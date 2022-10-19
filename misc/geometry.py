import cv2
import numpy as np
import torch 

from scipy.interpolate import interp1d
from skimage.draw import polygon, polygon_perimeter
from sympy.geometry.util import intersection, convex_hull
from sympy import Point, Polygon

from misc.log_utils import log

def project_to_ground_plane_pytorch(img, H, homography_input_size, homography_output_size, grounplane_img_size, padding_mode="zeros"):

    if len(img.shape) == 5:
        Bor, V = img.shape[:2]
        img = img.view(-1, *img.shape[2:])
        H = H.view(-1, *H.shape[2:])
    else:
        V = -1

    if list(img.shape[-2:]) != homography_input_size:
        #TODO resize
        # assert (np.array(org_input_size) != np.array(img.shape[-2:])).all()
        # #interpolate image to original sizz to make groundview projection correct
        log.spam("resising before projection")
        img = torch.nn.functional.interpolate(img, size=tuple(homography_input_size))

    #For proper alignement grounplane_img_size must match the value use to generate the homography
    B, C, h, w = img.shape

    h_grid, w_grid = homography_output_size
  
    yp_dist, xp_dist = torch.meshgrid(torch.arange(h_grid, device=img.device), torch.arange(w_grid, device=img.device))
    homogenous = torch.stack([xp_dist.float(), yp_dist.float(), torch.ones((h_grid, w_grid), device=img.device)]).reshape(1, 3, -1)
    
    if B != 1:
        homogenous = homogenous.repeat(B, 1, 1)

    map_ind  = H.bmm(homogenous)
    
    map_ind = (map_ind[:, :-1, :]/map_ind[:, -1, :].unsqueeze(1)).reshape(B, 2, h_grid, w_grid)
    map_ind = (map_ind / torch.tensor([w-1, h-1], device=img.device).reshape(1,2,1,1))*2 - 1

    grid = map_ind.permute(0,2,3,1)
    
    if padding_mode=="border":
        #Set the border to min value before projection and use min border for padding
        min_val = img.view(B, -1).min(dim=1).values.unsqueeze(1).unsqueeze(1)

        img[:,:,0,:] = min_val
        img[:,:,:,0] = min_val
        img[:,:,h-1,:] = min_val
        img[:,:,:,w-1] = min_val

    ground_image = torch.nn.functional.grid_sample(img, grid, mode='bilinear', align_corners=True,  padding_mode=padding_mode)

    ground_image[torch.isnan(ground_image)] = 0

    if grounplane_img_size != homography_output_size:
        #TODO resize
        log.spam("resising after projection")
        ground_image = torch.nn.functional.interpolate(ground_image, size=tuple(grounplane_img_size))

    if V != -1:
        ground_image = ground_image.view(Bor,V, *ground_image.shape[1:])
    # ground_image = ground_image_torch.squeeze().permute(1,2,0).numpy()

    return ground_image


def get_ground_plane_homography(K, R, T, world_origin_shift, grounplane_img_size, scale, height=0, grounplane_size_for_scale=None):

    #the if the scale is set with respect to particular grounplane size (grounplane_size_for_scale) and the homography has different ouput dimension
    #the scale is adjust such that the ouput is consistent no matter grounplane_img_size
    #in most cases scalex and scaley should be the same
    if grounplane_size_for_scale is not None and grounplane_img_size != grounplane_size_for_scale:
        scalex = scale * (grounplane_img_size[1] / grounplane_size_for_scale[1])
        scaley = scale * (grounplane_img_size[0] / grounplane_size_for_scale[0])
    else:
        scalex = scale
        scaley = scale

    Ki = np.array([[-scalex, 0, ((grounplane_img_size[1]-1)/2)], [0, scaley, ((grounplane_img_size[0]-1)/2)], [0, 0, 1]])

    T =  T + (R @ np.array([[world_origin_shift[0], world_origin_shift[1], height]]).T)
    
    RT = np.zeros((3,3))
    RT[:,:2] = R[:,:2]
    # RT[2,2] = 1
    RT[:,2] = T.squeeze()
    
    H = K @ RT @ np.linalg.inv(Ki)
    
    return H

def get_homograhy_from_corner_points(img_corner_square, img_floor_corner, ground_img_size):
    pts_dst = np.array([[0,0],
               [0,1],
               [1,1],
               [1,0]
              ], dtype=float)

    h, status = cv2.findHomography(img_corner_square, pts_dst)
    
    poing_hom = np.ones((img_floor_corner.shape[0], 3))
    poing_hom[:, :2] = img_floor_corner
    poing_hom = poing_hom.T
    
    groundview_points = h @ poing_hom
    groundview_points = (groundview_points[:-1] / groundview_points[-1]).T
    
    #normalize between zero and one
    groundview_points = groundview_points - np.min(groundview_points, axis=0)
    groundview_points = groundview_points / np.max(groundview_points, axis=0)
    
    #rescale to fit in ground_img_size
    groundview_points = groundview_points * np.expand_dims(np.array(ground_img_size), axis=0)
    
    h, status = cv2.findHomography(img_floor_corner, groundview_points)
    
    h = np.linalg.inv(h)

    return h

def project_points(points, homography):

    poing_hom = np.ones((points.shape[0], points.shape[1] + 1))
    poing_hom[:, :2] = points
    poing_hom = poing_hom.T

    projected_points = homography @ poing_hom
    projected_points = (projected_points[:-1] / projected_points[-1]).T

    return  projected_points
    
def project_image_points_to_groundview(points, ground_plane_homography):

    H_inv = np.linalg.inv(ground_plane_homography)

    poing_hom = np.ones((points.shape[0], points.shape[1] + 1))
    poing_hom[:, :2] = points
    poing_hom = poing_hom.T

    groundview_points = H_inv.dot(poing_hom)
    groundview_points = (groundview_points[:-1] / groundview_points[-1]).T

    return  groundview_points


def reproject_to_world_ground(ground_pix, K0, R0, T0):
    """
    Compute world coordinate from pixel coordinate of point on the groundplane
    """
    C0 = -R0.T @ T0
    l = R0.T @ np.linalg.inv(K0) @ ground_pix
    world_point = C0 - l*(C0[2]/l[2])
    
    return world_point
    
    
def project_world_to_camera(world_point, K1, R1, T1):
    """
    Project 3D point world coordinate to image plane (pixel coordinate)
    """
    point1 = ((R1 @ world_point) + T1)
    if(np.min(point1[2]) < 0 ):
        log.spam("Projection of world point located behind the camera plane")
    point1 = K1 @ point1
    point1 = point1 / point1[2]
    point1 = point1 / point1[2]
    
    return point1[:2]


def triangulate_point(points_2d, multi_calib):
    #Need at least point of view
    assert points_2d.shape[0] > 1
    
    #compute camera position for each view
    camera_positions = [-calib.R.T @ calib.T for calib in multi_calib]
    
    #Compute 3D direction from camera toward point
    point_directions = [-calib.R.T @ np.linalg.inv(calib.K) @ point for point, calib in zip(points_2d, multi_calib)]
    
    point_3d = nearest_intersection(np.array(camera_positions).squeeze(2), np.array(point_directions))
    
    return point_3d


def nearest_intersection(points, dirs):
    """
    :param points: (N, 3) array of points on the lines
    :param dirs: (N, 3) array of unit direction vectors
    :returns: (3,) array of intersection point
    
    from https://stackoverflow.com/questions/52088966/nearest-intersection-point-to-many-lines-in-python
    """
    #normalized direction
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs_mat = dirs[:, :, np.newaxis] @ dirs[:, np.newaxis, :]
    points_mat = points[:, :, np.newaxis]
    I = np.eye(3)
    return np.linalg.lstsq(
        (I - dirs_mat).sum(axis=0),
        ((I - dirs_mat) @ points_mat).sum(axis=0),
        rcond=None
    )[0]


def project_roi_world_to_camera(world_point, K1, R1, T1):
    """
    Project Region of interest 3D point world coordinate to image plane (pixel coordinate)
    A bit Hacky since world coordinate are sometime behind image plane, we interpolate between corner of polygon
    to only keep point in front of the image plane
    """

    point1 = ((R1 @ world_point) + T1)

    if point1[2].min() < 0:
        #If a corner point of the roi lie behind the image compute corespondence in the image plane
        x = world_point[0]
        y = world_point[1]

        # Evenly sample point around polygon define by corner point in world_point
        distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
        distance = distance/distance[-1]

        fx, fy = interp1d( distance, x ), interp1d( distance, y )

        alpha = np.linspace(0, 1, 150)
        x_regular, y_regular = fx(alpha), fy(alpha)

        world_point = np.vstack([x_regular, y_regular, np.zeros(x_regular.shape)])

        point1 = ((R1 @ world_point) + T1)
        
        #Filter out point behind the camera plane (Z < 0)    
        point1 = np.delete(point1, point1[2] < 0, axis=1)
    point1 = K1 @ point1
    point1 = point1 / point1[2]
    
    return point1[:2]


def update_img_point_boundary(img_points, view_ground_edge):
    #Make sure that all the img point are inside the image, if there are not replace them by points on the boundary
    img_points = map(Point, img_points)
    # img_corners = map(Point, [(0.0, 0.0), (0.0, img_size[0]), (img_size[1], img_size[0]), (img_size[1], 0.0)])
    img_corners = map(Point, view_ground_edge)

    poly1 = Polygon(*img_points)
    poly2 = Polygon(*img_corners)
    isIntersection = intersection(poly1, poly2)# poly1.intersection(poly2)
    
    point_inside = list(isIntersection)
    point_inside.extend([p for p in poly1.vertices if poly2.encloses_point(p)])
    point_inside.extend([p for p in poly2.vertices if poly1.encloses_point(p)])
   
    boundary_updated = convex_hull(*point_inside).vertices    
    boundary_updated = [p.coordinates for p in boundary_updated]

    return np.stack(boundary_updated).astype(float)


def project_world_point_to_groundplane(world_point, view_ground_edge, calib, H):
    """
    Project a point in 3d world coordinate lying on the ground (z=0), to it's 2D coordinate in the groundplane representation defined by world origin shift and rescale factor
    """
    
    #Project world coordinate to image plane (low res)
    point1 = ((calib.R @ world_point.T) + calib.T)
    point1 = calib.K @ point1

    point1 = point1 / point1[2]

    # point1 = point1 / point1[2]
    point1 = point1[:2].T
    
    #compute intersection with ROI in groundplane, and original image in groundplane
    
    if H is not None:
        #project point from image coordinate to groundplane
        if len(view_ground_edge) != 0:
            view_ground_edge = project_image_points_to_groundview(view_ground_edge, H)
        point1 = project_image_points_to_groundview(point1, H)
    #     point_ground = H @ point1
    if len(view_ground_edge) != 0:
        point1 = update_img_point_boundary(point1, view_ground_edge)
    
    return point1


def generate_roi_and_boundary_from_corner_points(hm_size, org_size, corner_points, view_ground_edge, calib, H, is_world_coordinate=False, image_plane=False):
    """
    Generate regions of interest and region boundary from a set of corner point, if is_world_coordinate is true, 
    it assumes corner_points are 3d world cordinate and first project them to their 2d grounplane coordinate.
    """
    corner_points = np.array(corner_points)

    if is_world_coordinate:
        #convert world point to point in ground plane
        corner_points = project_world_point_to_groundplane(corner_points, view_ground_edge, calib, H)

    if image_plane:
        #if working in the image plane downscale roi point to be in the same scale as heatmaps
        corner_points = corner_points / 8

    poly = np.array(corner_points)
    # poly = np.around(poly)

    mask_boundary = np.zeros(hm_size)
    rr, cc = polygon_perimeter(poly[:,1], poly[:,0], mask_boundary.shape)
    mask_boundary[rr,cc] = 1
    mask_boundary = mask_boundary

    roi = np.zeros(hm_size)
    # we put mask boundary into ROI to make sure they overlap
    # roi[rr,cc] = 1
    rr, cc = polygon(poly[:,1], poly[:,0], roi.shape)
    roi[rr,cc] = 1

    return roi, mask_boundary


def remove_occluded_area_from_mask(mask, second_mask, occluded_area_list, view_ground_edge, org_size, calib, H, is_world_coordinate=True, image_plane=False):
    """
    mask: array to edit
    occluded_area_list: a list where each element corespond to an occluded area. each occluded area is representedn by its corner points.
    """

    for occluded_area in occluded_area_list:
        corner_points = np.array(occluded_area)

        if is_world_coordinate:
            #convert world point to point in ground plane
            corner_points = project_world_point_to_groundplane(corner_points, view_ground_edge, calib, H)

        if image_plane:
            #if working in the image plane downscale roi point to be in the same scale as heatmaps
            corner_points = corner_points / 8

        poly = np.array(corner_points)
        rr, cc = polygon(poly[:,1], poly[:,0], mask.shape)
        mask[rr,cc] = 0
        second_mask[rr,cc] = 0


    return mask, second_mask


def update_K_after_resize(K, old_size, new_size):
    fx = 1.0 / (old_size[1] / new_size[1])
    fy = 1.0 / (old_size[0] / new_size[0])

    scaler = np.array([
        [fx, 0, 0],
        [0, fy, 0],
        [0, 0, 1]]
    )

    new_K = scaler @ K

    return new_K

def rescale_keypoints(points, org_img_dim, out_img_dim):

    if len(points) == 0:
        return points
    
    out_img_dim = np.array(out_img_dim)
    org_img_dim = np.array(org_img_dim)
    
    if np.all(org_img_dim == out_img_dim):
        return points
    
    resize_factor = out_img_dim / org_img_dim
    #swap x and y
    resize_factor = resize_factor[::-1]

    resized_points = points*resize_factor

    return resized_points

def distance_point_to_line(lp1, lp2, p3):
    #Both point of the line are the same return distance to that point
    if np.all(lp1 == lp2):
        return np.linalg.norm(p3-lp1)
    else:
        return np.abs(np.cross(lp2-lp1, lp1-p3) / np.linalg.norm(lp2-lp1))