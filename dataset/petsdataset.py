import json
import numpy as np

from collections import defaultdict, namedtuple
from pathlib import Path
from scipy.spatial.transform import Rotation
from xml.dom import minidom

from dataset.utils import Calibration, Bbox, Annotations, find_nearest_point, get_frame_from_file
from dataset.sceneset import SceneBaseSet
from misc import geometry
from misc.log_utils import log



class PETSSceneSet(SceneBaseSet):
    def __init__(self, data_conf, scene_config_file, Seq, Lset, Tstmp, view_subset=[1,2,3,4,5,6,7,8]):
        super().__init__(data_conf, scene_config_file)
        
        self.root = Path(self.scene_config["data_root"])
        self.video_path = self.root / "Crowd_PETS09/S{}/L{}/Time_{:02d}-{:02d}".format(Seq, Lset, *Tstmp)
        self.gt_path = self.root / "gt/PETS2009-S{}L{}_{:02d}-{:02d}.xml".format(Seq, Lset, *Tstmp) 
     
        self.label_path = "./data/PETS2009/labels/S{}L{}/{:02d}_{:02d}/GP_pmap_height.json".format(Seq, Lset, *Tstmp)
        
        self.available_views = get_PETS_available_views(self.video_path, view_subset)
        
        assert self.available_views[0] == 1
        
        #Some frame are missing in certain view
        #TODO cap nb frame to the view with least frame?
        self.nb_frames = min([len([frame_path for frame_path in (self.video_path / f'View_00{view_id}/').iterdir() if frame_path.suffix == ".jpg"]) for view_id in self.available_views])
        
        self.calibs = load_PETS_multi_calibration(self.root, view_list=self.available_views)
        
        self.world_origin_shift = self.scene_config["world_origin_shift"]
        self.groundplane_scale = self.scene_config["grounplane_scale"]

        self.gt, set_name = load_PETS_gt(self.gt_path)
        # self.label_gt = load_PETS_label(self.label_path)
        
        assert len(self.gt) >= self.nb_frames

        # #Manual head label are missing on some frame
        # if self.nb_frames != len(self.label_gt):
        #     log.warning(f"Head labels are missing for {self.nb_frames - len(self.label_gt)} frames")    
        
        log.debug(f"Dataset {set_name} containing {self.nb_frames} frames from {self.get_nb_view()} views with camera name: {self.available_views}")

        #use parent class to generate ROI and occluded area maps
        self.generate_scene_elements()
        self.log_init_completed()

    def get_frame(self, index, view_id):
        """
        Read and return undistoreted frame coresponding to index and view_id.
        The frame is return at the original resolution
        """
        frame_path = Path(self.video_path) / 'View_{0:03d}'.format(self.available_views[view_id]) / 'frame_{0:04d}.jpg'.format(index)

        # log.debug(f"pomelo dataset get frame {index} {view_id}")
        frame = get_frame_from_file(frame_path)

        return frame

    def _get_gt(self, index, view_id):
        
        gt = extend_PETS_gt(self.gt[index], self.calibs)[view_id] #self.label_gt[index]

        return gt

    def _get_homography(self, view_id):
        """
        return the homography projecting the image to the groundplane.
        It takes into account potential resizing of the image and
        uses class variables:
         frame_original_size, homography_input_size, homography_output_size,
        as parameters
        """

        # update camera parameter to take into account resizing from the image before homography is applied
        K = geometry.update_K_after_resize(self.calibs[view_id].K, self.frame_original_size, self.homography_input_size)
        H = geometry.get_ground_plane_homography(K, self.calibs[view_id].R, self.calibs[view_id].T, self.world_origin_shift, self.homography_output_size, self.groundplane_scale, grounplane_size_for_scale=[128,128])
        
        return H
        
    def get_nb_view(self):
        return len(self.calibs)

    def get_length(self):
        return self.nb_frames


def load_PETS_single_calibration(path):

    """
    Tsai calibration conversion to KRT following doc below
    http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
    http://campar.in.tum.de/twiki/pub/Far/AugmentedRealityIntroductionWiSe2003/L10-Jan20-CamCalib.pdf
    """
    

    Geometric = namedtuple('Geometric', ['w', 'h', 'ncx', 'nfx', 'dx', 'dy', 'dpx', 'dpy'])
    Intrinsic = namedtuple('Intrinsic', ['f', 'kappa1', 'cx', 'cy', 'sx'])
    Extrinsic = namedtuple('Exctrinsic', ['tx', 'ty', 'tz', 'rx', 'ry', 'rz'])

    gt_xml = minidom.parse(path)

    camera = gt_xml.getElementsByTagName('Camera')

    assert len(camera) == 1

    camera_name = camera[0].attributes['name'].value

    #Tsai calibration parameters
    geom = gt_xml.getElementsByTagName('Geometry')[0]
    intr = gt_xml.getElementsByTagName('Intrinsic')[0]
    extr = gt_xml.getElementsByTagName('Extrinsic')[0]

    geom = Geometric(w=int(geom.attributes["width"].value), h=int(geom.attributes["height"].value), ncx=float(geom.attributes["ncx"].value), nfx=float(geom.attributes["nfx"].value), dx=float(geom.attributes["dx"].value), dy=float(geom.attributes["dy"].value), dpx=float(geom.attributes["dpx"].value), dpy=float(geom.attributes["dpy"].value))
    intr = Intrinsic(f=float(intr.attributes["focal"].value), kappa1=float(intr.attributes["kappa1"].value), cx=float(intr.attributes["cx"].value), cy=float(intr.attributes["cy"].value), sx=float(intr.attributes["sx"].value))
    extr = Extrinsic(tx=float(extr.attributes["tx"].value), ty=float(extr.attributes["ty"].value), tz=float(extr.attributes["tz"].value), rx=float(extr.attributes["rx"].value), ry=float(extr.attributes["ry"].value), rz=float(extr.attributes["rz"].value))

    #Rebuild K, R, T matrices/vector
    K = np.array([[(intr.sx/geom.dpx)*intr.f, 0, intr.cx], [0, (1/geom.dpy)*intr.f, intr.cy], [0, 0, 1]])
    R = Rotation.from_euler('xyz', [extr.rx, extr.ry, extr.rz], degrees=False).as_matrix()
    T = np.array([[extr.tx, extr.ty, extr.tz]]).T

    return K, R, T


def load_PETS_multi_calibration(root_path, view_list=[1,2,3,4,5,6,7,8]):
    
    calib_list = list()
    for view_id in view_list:
        
        camera_matrix_path = Path(root_path) / 'cparsxml/View_{0:03d}.xml'.format(view_id)
        
        K, R, T = load_PETS_single_calibration(str(camera_matrix_path))
        
        #number view starting from zero
        calib_list.append(Calibration(K=K, R=R, T=T, view_id=view_id))
        
    return calib_list


def load_PETS_gt(gt_path):    
    
    gt_xml = minidom.parse(str(gt_path))

    dataset = gt_xml.getElementsByTagName('dataset')
    
    assert len(dataset) == 1
    
    set_name = dataset[0].attributes['name'].value

    frames = [x for x in dataset[0].childNodes if x.nodeType == minidom.Node.ELEMENT_NODE]
    
    bbox_per_frame = defaultdict(list)
    for frame in frames:
        frame_number = int(frame.attributes["number"].value)
        
        bbox_list = [x for x in frame.childNodes if x.nodeType == minidom.Node.ELEMENT_NODE]
        assert len(bbox_list) == 1
        bbox_list = [x for x in bbox_list[0].childNodes if x.nodeType == minidom.Node.ELEMENT_NODE]
        
        for bbox in bbox_list:
            person_id = int(bbox.attributes['id'].value)

            bbox = [x for x in bbox.childNodes if x.nodeType == minidom.Node.ELEMENT_NODE]
            assert len(bbox) == 1
            bbox = Bbox(xc=float(bbox[0].attributes["xc"].value), yc=float(bbox[0].attributes["yc"].value), w=float(bbox[0].attributes["w"].value), h=float(bbox[0].attributes["h"].value))
            
            bbox_per_frame[frame_number].append((bbox, person_id, frame_number))
    
    return bbox_per_frame, set_name


def load_PETS_label(gt_path):

    """
    load head annotation from citystreet dataset 
    """
    with open(gt_path, 'r') as _f:
        label_json = json.load(_f)
    
    label_per_frame = defaultdict(list)
    for frame, annotation in label_json.items():
        frame_number = int(frame[6:10])
        
        for person_id, point in annotation["regions"].items():
            point = point["shape_attributes"]
            
            feetw = correct_world_coord_citysstreet([point["cx"], point["cy"], 0])
            headw = correct_world_coord_citysstreet([point["cx"], point["cy"], point["height"]])

            label_per_frame[frame_number].append((feetw, headw, point["height"]))
    
    return label_per_frame


def correct_world_coord_citysstreet(world_point):
    bbox = [-31, 29, -45, 25] 

    world_point = np.array(world_point) / 10.0
    world_point = world_point + np.array([-31, -45, 0])

    world_point = world_point * np.array([1000, 1000, 10])


    world_point = world_point 
    
    return world_point.reshape(3,1)

def project_feet(left_bottom, right_bottom, center_bottom, K0, R0, T0, K1, R1, T1):
    
    #reprojection of bbox bottom
    w_left = geometry.reproject_to_world_ground(left_bottom, K0, R0, T0)
    w_right = geometry.reproject_to_world_ground(right_bottom, K0, R0, T0)
    w_center = geometry.reproject_to_world_ground(center_bottom, K0, R0, T0)
    
    # camera orientation projected on ground plane
    C0 = (R0.T @ T0)
    C0[2] = 0
    C0 = C0 / np.linalg.norm(C0)

    #center of the bounding box cube/cylinder
    feet_world = w_center + C0*np.linalg.norm((w_right-w_left)/2)
    
    feet_reproj = geometry.project_world_to_camera(feet_world, K1, R1, T1)
    
    return feet_reproj, feet_world
    

def project_head(feet_world, center_top, K0, R0, T0, K1=None, R1=None, T1=None):
    
    #Camera position
    C0 = -R0.T @ T0 
    
    #direction from camera to bbox top
    l = -R0.T @ np.linalg.inv(K0) @ center_top
    
    # plane orthogonal to ground going through feet and facing camera
    p0 = feet_world
    n = C0.copy()
    n[2] = 0 
    n = n / np.linalg.norm(n)
    
    #intersection between plane and line from camera to bbox top
    d = (p0 - C0).squeeze().dot(n.squeeze()) / l.squeeze().dot(n.squeeze())
    head_world = C0 + l*d
    
    if K1 is not None:
        head_reproj = geometry.project_world_to_camera(head_world, K1, R1, T1)
    else:
        head_reproj = None
    
    return head_reproj, head_world


def extend_PETS_gt(gt, calib_multi, label_gt=None):
    
    """
    Add estimated position of feet and head derived from bounding box 
    Project position feet/head position from view 0 to the other view using camera calibratiion
    """
                        
    K0 = calib_multi[0].K
    R0 = calib_multi[0].R
    T0 = calib_multi[0].T
    
    gts_multi = list()
    for calib in calib_multi:
    
        K1 = calib.K
        R1 = calib.R
        T1 = calib.T
        
        view_gts = list()
        
        for bbox, person_id, frame_number in gt:
            bbox_bottom_center = np.array([[bbox.xc], [bbox.h / 2.0 + bbox.yc], [1]])
            bbox_bottom_left = np.array([[bbox.xc - bbox.w / 2.0], [bbox.h / 2.0 + bbox.yc], [1]])
            bbox_bottom_right = np.array([[bbox.xc + bbox.w / 2.0], [bbox.h / 2.0 + bbox.yc], [1]])

            feet_reproj, feet_world = project_feet(bbox_bottom_left, bbox_bottom_right, bbox_bottom_center, K0, R0, T0, K1, R1, T1)

            if label_gt is None or len(label_gt) == 0:
                #Head annotation missing use bounding box extrapolation
                bbox_top_center = np.array([[bbox.xc], [-bbox.h / 2.0 + bbox.yc], [1]])
                head_reproj, head_world = project_head(feet_world, bbox_top_center, K0, R0, T0, K1, R1, T1)
                height = np.linalg.norm(head_world[1]-feet_world[1])
                
            else:
                nearest_id = find_nearest_point(feet_world, [lab[0] for lab in label_gt])
                feet_reproj = geometry.project_world_to_camera(label_gt[nearest_id][0], K1,R1,T1)
                head_reproj = geometry.project_world_to_camera(label_gt[nearest_id][1], K1,R1,T1)
                height = label_gt[nearest_id][2]
            
            if calib.view_id == 1:
                #original view we keep the bounding box
                bbox = bbox
            else:
                bbox = None
            
            view_gts.append(Annotations(bbox=bbox, head=head_reproj, feet=feet_reproj, height=height, id=person_id, frame=frame_number, view=calib.view_id))
    
        gts_multi.append(view_gts)

    return gts_multi
    

def get_PETS_available_views(video_path, view_subset):
    available_view_ids = list()               
    for path in video_path.iterdir():
        if path.name[:5] == "View_":
            view_id = int(path.name[5:8])
            if view_id in view_subset:
                available_view_ids.append(view_id)
                
    return sorted(available_view_ids)


def get_scene_set(dataset_root_path):

    scene_set_list = list()

    for S, L, T in PET_SCENE_SET:
        scene_set_list.append(PETSSceneSet(dataset_root_path, S, L, T, [1,2,3,4,5,6,7,8]))

    return scene_set_list


def get_specific_scene_set(dataset_root_path, S, L, T):
    return PETSSceneSet(dataset_root_path, S, L, T, [1,2,3,4,5,6,7,8])
