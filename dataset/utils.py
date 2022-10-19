import os
import time

import cv2
import json
import numpy as np
import torch 


from collections import namedtuple, defaultdict
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
from skimage.draw import polygon, polygon_perimeter

from misc.geometry import  rescale_keypoints, project_image_points_to_groundview
from misc.log_utils import log

#Commonly use namedtuple to encapsulate basic data
Calibration = namedtuple('Calibration', ['K', 'R', 'T', 'view_id'])
Bbox = namedtuple('Bbox', ['xc', 'yc', 'w', 'h']) #, 'id', 'frame'])
Annotations = namedtuple('Annotations', ['bbox', 'head', 'feet', 'height', 'id', 'frame', 'view'])
Homography = namedtuple('Homography', ['H', 'input_size', 'output_size'])


def is_in_frame(point, frame_size):
    is_in_top_left = point[0] > 0 and point[1] > 0
    is_in_bottom_right = point[0] < frame_size[0] and point[1] < frame_size[1]
    
    return is_in_top_left and is_in_bottom_right


def check_if_in_roi(roi, point):
    #if point is completely ou of roi return false
    if point[0] >= roi.shape[-1] or point[1] >= roi.shape[-2] or point[0] < 0 or point[1] < 0:
        return False
    
    return roi[point[1], point[0]] == 1

def find_nearest_point(point, point_list):
    
    point_list = np.asarray(point_list)
    dist_2 = np.sum((point_list - point)**2, axis=1)
    
    return np.argmin(dist_2)

def interpolate_annotations(ann_dict, index):
    
    existing_index = sorted(list(ann_dict.keys()))
    
    ind_pos = np.searchsorted(existing_index, index)

    if ind_pos >= len(existing_index):
        return []

    before = existing_index[ind_pos - 1]
    after = existing_index[ind_pos]
    
    inter_weight = (index - before) / (after - before)
    
    ann_interpolated = list()

    for ann in ann_dict[before]:
        ann_after = [ann_after for ann_after in ann_dict[after] if ann_after.id == ann.id]
        if len(ann_after) > 0:
            assert len(ann_after) == 1

            ann_after = ann_after[0]
            
            if ann.head is not None and ann_after.head is not None:
                head_reproj = ann.head*(1-inter_weight) + ann_after.head*inter_weight
            else:
                head_reproj = None
            
            if ann.height is not None and ann_after.height is not None:
                height = ann.height*(1-inter_weight) + ann_after.height*inter_weight
            else:
                height = None

            #is feet is not visible in frame before or after we remove annotation
            if ann.feet is not None and ann_after.feet is not None:  
                feet_reproj = ann.feet *(1-inter_weight)+ ann_after.feet*inter_weight

                ann_interpolated.append(Annotations(bbox=None, head=head_reproj,  feet=feet_reproj, height=height, id=ann.id, frame=index, view=ann.view))

    return ann_interpolated

def undistort_gt(ann_list, K, dist_coeff):
    undi_anns = list()
    for ann in ann_list:
        
        if ann.head is not None:
            head_undi = cv2.undistortPoints(ann.head, K, dist_coeff, P=K).reshape(2,1)
        else:
            head_undi = ann.head

        if ann.feet is not None:
            feet_undi = cv2.undistortPoints(ann.feet, K, dist_coeff, P=K).reshape(2,1)
        else:
            feet_undi = ann.feet
        
        undi_anns.append(ann._replace(head=head_undi, feet=feet_undi))

    return undi_anns


def resize_density(hm, hm_size, scale_factor=None, mode="bilinear"):
    '''
    Resize heatmap and rescale value in order to keep density consitent    
    '''
    
    if np.all(np.array(hm.shape[-2:]) == np.array(hm_size)):
        # Size already correct
        return hm

    hm_resized = torch.nn.functional.interpolate(hm, size=tuple(hm_size), mode=mode)

    if scale_factor is None:
        scale_factor = hm.sum() / hm_resized.sum()
        # scale_factor = np.prod(np.array(hm.shape[-2:]) / np.array(hm_size))

    #Scale heatmap such that density sum is equal after resizing
    hm_resized = hm_resized * scale_factor

    return hm_resized


def read_json_file(filepath):
    with open(os.path.abspath(filepath)) as f:    
        json_dict = json.load(f)

    return json_dict

def read_sloth_annotation(ann_pathes):
    
    scene_ann = []
    
    person_id_set = set()
    for view_id, ann_view in enumerate(ann_pathes):
        
        view_ann = dict()
        
        data = read_json_file(ann_view)
        
        print(f"Scene {view_id+1} containing {len(data)} frames")
        
        for frame_json in data:
            frame_id = int(frame_json["filename"].split("/")[-1][12:-4])
            frame_ann = defaultdict(dict)
            
            for point_ann in frame_json["annotations"]:
                person_id_set.add(int(point_ann["person_id"]))
                frame_ann[int(point_ann["person_id"])][point_ann["class"]] =  (point_ann["x"], point_ann["y"])
            
            view_ann[frame_id] = frame_ann
            
        scene_ann.append(view_ann)
    
    return scene_ann, list(person_id_set)


def get_frame_from_file(frame_path):
    
    if not(frame_path.is_file()):
        log.error(f"Trying to load {frame_path} which is not a file")
        assert frame_path.is_file()

    frame = cv2.imread(str(frame_path))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame  


def get_frame_from_video(video_path, frame_index):
    while True:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        res, frame = cap.read()
        
        cap.release()
        
        if res and frame is not None:
            break

        time.sleep(1)

    return frame   


def get_train_val_split(scene_set_cumsum_lengths, split_proportion):
    train_idx = list()
    test_idx = list()

    for i in range(len(scene_set_cumsum_lengths)):
        set_end = scene_set_cumsum_lengths[i]

        if i != 0:
            prev_end = scene_set_cumsum_lengths[i-1]
        else:
            prev_end = 0
        
        train_size = int((set_end - prev_end) * split_proportion)

        train_idx.extend(list(range(prev_end, prev_end + train_size)))
        test_idx.extend(list(range(prev_end + train_size, set_end)))
    
    return train_idx, test_idx

def get_train_val_split_index(dataset, split_proportion):
    train_size = int(len(dataset) * split_proportion)

    train_idx = list(range(0, train_size))
    test_idx = list(range(train_size, len(dataset)))

    return train_idx, test_idx


def generate_scene_roi_from_view_rois(view_ROIs, view_homographies, frame_original_size, homography_input_size, homography_output_size, hm_size):
    """
    given region of interest (defines as points on the perimeter of the the region) generate 
    the region of interest and it's boundary for the full scene in the grounplane caracterised by the homographies.
    """

    #rescame0 ROI corner point in image to match homography size
    if frame_original_size != homography_input_size:
        view_ROIs = [rescale_keypoints(roi_v, frame_original_size, homography_input_size)  for roi_v in view_ROIs]

    view_ROIs_ground = [project_image_points_to_groundview(img_roi, homography) for img_roi, homography in zip(view_ROIs, view_homographies)]

    #rescale ROI corner in grounplane to match output hm size
    if homography_output_size != hm_size:
        view_ROIs_ground =  [rescale_keypoints(roi_ground, homography_output_size, hm_size)  for roi_ground in view_ROIs_ground]

    #merge the polygons from each view ROI into a single polygon
    polygons = [Polygon(roi_ground) for roi_ground in view_ROIs_ground]

    try:
        ROI_ground_poly = np.array(list(unary_union(polygons).exterior.coords))
    except:
        log.warning("ROI couldn't be generated, returning ROI covering the full groundplane")
        return np.ones(hm_size), np.zeros(hm_size)

    roi = np.zeros(hm_size)
    mask_boundary = np.zeros(hm_size)

    rr, cc = polygon_perimeter(ROI_ground_poly[:,1], ROI_ground_poly[:,0], mask_boundary.shape, clip=True)
    mask_boundary[rr,cc] = 1
    mask_boundary = mask_boundary

    roi[rr,cc] = 1
    rr, cc = polygon(ROI_ground_poly[:,1], ROI_ground_poly[:,0], roi.shape)
    roi[rr,cc] = 1

    return roi, mask_boundary


def aggregate_multi_view_gt(anns_gt, homographies, frame_original_size, homography_input_size, homography_output_size, hm_size):
        feets_ground = defaultdict(lambda: defaultdict(list))

        #Apply groundplane homography to all annotation from differents view
        for view_id in range(len(homographies)):
            anns_view = anns_gt[view_id]
            h_view = homographies[view_id]

            for ann in anns_view:
                
                feet_loc = ann.feet.reshape(1,2)

                #rescale input point if they don't match homography input size
                if frame_original_size != homography_input_size:
                    feet_loc =  rescale_keypoints(feet_loc, frame_original_size, homography_input_size)

                feet_ground = project_image_points_to_groundview(feet_loc, h_view)

                #rescale feet point in the ground according to final heatmap dimension
                if homography_output_size != hm_size:
                    feet_ground =  rescale_keypoints(feet_ground, homography_output_size, hm_size)

                feets_ground[ann.frame][ann.id].append(feet_ground.squeeze())

        #Aggregate annotation and generate final groundplane annoatation
        anns_ground = list()
        for frame_id, frame_anns in feets_ground.items():
            for person_id, feet_ground_list in frame_anns.items():
                feet_ground_agg = np.mean(np.array(feet_ground_list), axis=0)
                anns_ground.append(Annotations(bbox=None, head=None,  feet=feet_ground_agg, height=None, id=person_id, frame=frame_id, view="ground"))

        return anns_ground


def extract_points_from_gt(gt, hm_size, gt_original_size=None):
        '''
        extract point and person id from a list of gt named tuple
        '''

        if len(gt) == 0:
            return gt, []

        #Filter visible point in that view
        gt_points = np.array([ann.feet for ann in gt if ann.feet is not None])
        person_id = np.array([ann.id for ann in gt if ann.feet is not None])

        if len(gt_points.shape) > 2:
            gt_points = gt_points.squeeze(2)

        if gt_original_size is not None and gt_original_size != hm_size:
            assert len(gt_points.shape) <= 2, f"gt points should have two dimensions: {gt_points.shape}"
            gt_points =  rescale_keypoints(gt_points, gt_original_size, hm_size)

        #Filter out point outside of hm size
        mask_visible = np.array([is_in_frame(point[[1,0]], hm_size) for point in gt_points])
        gt_points = gt_points[mask_visible] 
        person_id = person_id[mask_visible]

        return gt_points, person_id

def aggregate_multi_view_gt_points(gt_points, gt_person_id, homographies, frame_original_size, homography_input_size, homography_output_size, hm_size):
        """
        Takes a list of list of points and person id, and a list of homographies.
        Project the list of each view in a common groundplane using homographies, and aggregate projection to generate gt in grounplane
        """
        points_ground = defaultdict(list)

        #Apply groundplane homography to all annotation from differents view
        for view_id in range(len(homographies)):
            gt_view = gt_points[view_id]
            id_view = gt_person_id[view_id]
            h_view = homographies[view_id]

            for point, person_id in zip(gt_view, id_view):
                
                point = point.reshape(1,2)

                #rescale input point if they don't match homography input size
                if frame_original_size != homography_input_size:
                    point =  rescale_keypoints(point, frame_original_size, homography_input_size)

                point_ground = project_image_points_to_groundview(point, h_view)

                #rescale feet point in the ground according to final heatmap dimension
                if homography_output_size != hm_size:
                    point_ground =  rescale_keypoints(point_ground, homography_output_size, hm_size)

                points_ground[person_id].append(point_ground.squeeze())

        #Aggregate annotation and generate final groundplane annoatation
        agg_points_ground = list()
        for person_id, feet_ground_list in points_ground.items():
            feet_ground_agg = np.mean(np.array(feet_ground_list), axis=0)

            if is_in_frame(feet_ground_agg[[1,0]], hm_size):
                agg_points_ground.append((feet_ground_agg, person_id))

        if len(agg_points_ground) == 0:
            return [], []
            
        gt_points_ground, gt_ground_person_id = [np.array(x) for x in zip(*agg_points_ground)]

        return gt_points_ground, gt_ground_person_id

def get_flow_channel(x, y, x_prev, y_prev):

    #deal with radius larger than one:
    if x_prev > x:
        x_prev = x + 1
    elif x_prev < x:
        x_prev = x - 1

    if y_prev > y:
        y_prev = y + 1
    elif y_prev < y:
        y_prev = y - 1

    if x == x_prev and y == y_prev:
        return 4
    if x == x_prev and y == y_prev+1:
        return 7
    if x == x_prev+1 and y == y_prev:
        return 5
    if x == x_prev+1 and y == y_prev+1:
        return 8
    if x == x_prev and y == y_prev-1:
        return 1
    if x == x_prev-1 and y == y_prev:
        return 3
    if x == x_prev-1 and y == y_prev-1:
        return 0
    if x == x_prev+1 and y == y_prev-1:
        return 2
    if x == x_prev-1 and y == y_prev+1:
        return 6

def generate_motion_tuple(pre_gt, pre_gt_person_id, gt, gt_person_id):
    
    if len(pre_gt) != 0:
        pre_gt = np.rint(pre_gt).astype(int)

    if len(gt) != 0:
        gt = np.rint(gt).astype(int)
    
    if not isinstance(pre_gt_person_id, list):
        pre_gt_person_id = pre_gt_person_id.tolist()
    if not isinstance(gt_person_id, list):
        gt_person_id = gt_person_id.tolist()

    # only keep person id present in both timestep
    common_person_id = set(pre_gt_person_id).intersection(gt_person_id)

    #create tuple of positions for each person present both in pre_gt and gt
    common_position = [(pre_gt[pre_gt_person_id.index(p_id)], gt[gt_person_id.index(p_id)]) for p_id in common_person_id]        
    
    return common_position
    
def generate_flow(pre_gt, pre_gt_person_id, gt, gt_person_id, roi, hm_radius, generate_hm=True):
    
    if generate_hm:
        gt_flow = np.zeros((10, roi.shape[-2], roi.shape[-1]))
    else:
        gt_flow = None

    roi = roi.squeeze()

    common_position = generate_motion_tuple(pre_gt, pre_gt_person_id, gt, gt_person_id) 
    
    gt_flow_tuple = list()
    move_length = list()
    for (pres_pos, pos) in common_position:
        nb_pixel_move = cdist([pres_pos], [pos], 'chebyshev').item()
        move_length.append(nb_pixel_move)
        if  nb_pixel_move <= 1:
            if check_if_in_roi(roi, pres_pos):
                flow_channel = get_flow_channel(pos[0], pos[1], pres_pos[0], pres_pos[1])
                # log.debug(f"{pres_pos}, {pos}, {flow_channel}")
                if generate_hm:
                    cv2.circle(gt_flow[flow_channel], tuple(pres_pos), hm_radius, 1, thickness=cv2.FILLED)
                gt_flow_tuple.append((pres_pos, flow_channel))
            elif check_if_in_roi(roi, pos):
                #incoming flow from the outside of roi
                if generate_hm:
                    cv2.circle(gt_flow[9], tuple(pos), hm_radius, 1, thickness=cv2.FILLED)
                gt_flow_tuple.append((pos, 9))

    if generate_hm:
        gt_flow = torch.from_numpy(gt_flow).to(torch.float32)
        
    
    return gt_flow, gt_flow_tuple, move_length