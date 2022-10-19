import os
import json

import cv2
import numpy as np
import xml.etree.ElementTree as ElementTree

from collections import defaultdict
from pathlib import Path
from xml.dom import minidom

from dataset.utils import Calibration, Bbox, Annotations, get_frame_from_file
from dataset.sceneset import SceneBaseSet
from misc import geometry
from misc.log_utils import log

class WildtrackSet(SceneBaseSet):
    def __init__(self, data_conf, scene_config_file):
        super().__init__(data_conf, scene_config_file)
        
        self.root_path = Path(self.scene_config["data_root"])    

        self.frame_dir_path = self.root_path / "Image_subsets/"
        self.gt_dir_path = self.root_path / "annotations_positions/"

        self.nb_frames = len([frame_path for frame_path in (self.gt_dir_path).iterdir() if frame_path.suffix == ".json"])
        
        self.calibs = load_calibrations(self.root_path)

        self.world_origin_shift = self.scene_config["world_origin_shift"]
        self.groundplane_scale = self.scene_config["grounplane_scale"]

        log.debug(f"Dataset Wildtrack containing {self.nb_frames} frames from {self.get_nb_view()} views ")

        #use parent class to generate ROI and occluded area maps
        self.generate_scene_elements()
        self.log_init_completed()

    def get_frame(self, index, view_id):
            """
            Read and return undistoreted frame coresponding to index and view_id.
            The frame is return at the original resolution
            """
            index = index * 5 

            frame_path = self.frame_dir_path / "C{:d}/{:08d}.png".format(view_id + 1, index)

            # log.debug(f"pomelo dataset get frame {index} {view_id}")
            frame = get_frame_from_file(frame_path)

            return frame

    def _get_gt(self, index, view_id):
        
        index = index * 5 

        gt_path = self.gt_dir_path / "{:08d}.json".format(index)
        gt = read_json(gt_path, self.calibs)[view_id]

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


class WildtrackExtendedSet(SceneBaseSet):
    def __init__(self, data_conf, scene_config_file):
        super().__init__(data_conf, scene_config_file)
        
        self.root_path = Path(self.scene_config["data_root"])    

        self.frame_dir_path = self.root_path / "Image_subsets/"
        self.gt_dir_path = self.root_path / "annotations_positions/"

        self.frame_dir_extended_path = self.root_path / "Image_extended_subsets/"
        self.gt_dir_extended_path = self.root_path / "annotations_extended_positions"

        self.nb_frames = len([frame_path for frame_path in (self.gt_dir_path).iterdir() if frame_path.suffix == ".json"]) + len([frame_path for frame_path in (self.gt_dir_extended_path).iterdir() if frame_path.suffix == ".json"]) - 1
        
        self.calibs = load_calibrations(self.root_path)

        self.world_origin_shift = self.scene_config["world_origin_shift"]
        self.groundplane_scale = self.scene_config["grounplane_scale"]

        log.debug(f"Dataset Wildtrack containing {self.nb_frames} frames from {self.get_nb_view()} views ")

        #use parent class to generate ROI and occluded area maps
        self.generate_scene_elements()
        self.log_init_completed()

    def get_frame(self, index, view_id):
            """
            Read and return undistoreted frame coresponding to index and view_id.
            The frame is return at the original resolution
            """
            #start from frame 5 frame between 0 and 5 are not available
            index = index + 5  

            if index % 5 == 0:
                frame_path = self.frame_dir_path / "C{:d}/{:08d}.png".format(view_id + 1, index)
            else:
                frame_path = self.frame_dir_extended_path / "C{:d}/{:08d}.png".format(view_id + 1, index)


            # log.debug(f"pomelo dataset get frame {index} {view_id}")
            frame = get_frame_from_file(frame_path)

            return frame

    def _get_gt(self, index, view_id):
        
        index = index + 5 

        if index % 5 == 0:
            gt_path = self.gt_dir_path / "{:08d}.json".format(index)
        else:
            gt_path = self.gt_dir_extended_path / "{:08d}.json".format(index)

        gt = read_json(gt_path, self.calibs)[view_id]

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


def load_opencv_xml(filename, element_name, dtype='float32'):
    """
    Loads particular element from a given OpenCV XML file.
    Raises:
        FileNotFoundError: the given file cannot be read/found
        UnicodeDecodeError: if error occurs while decoding the file
    :param filename: [str] name of the OpenCV XML file
    :param element_name: [str] element in the file
    :param dtype: [str] type of element, default: 'float32'
    :return: [numpy.ndarray] the value of the element_name
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError("File %s not found." % filename)
    try:
        tree = ElementTree.parse(filename)
        rows = int(tree.find(element_name).find('rows').text)
        cols = int(tree.find(element_name).find('cols').text)
        return np.fromstring(tree.find(element_name).find('data').text,
                             dtype, count=rows*cols, sep=' ').reshape((rows, cols))
    except Exception as e:
        print(e)
        raise UnicodeDecodeError('Error while decoding file %s.' % filename)
        

def load_all_extrinsics(_lst_files):
    """
    Loads all the extrinsic files, listed in _lst_files.
    Raises:
        FileNotFoundError: see _load_content_lines
        ValueError: see _load_content_lines
    :param _lst_files: [str] path of a file listing all the extrinsic calibration files
    :return: tuple of ([2D array], [2D array]) where the first and the second integers
             are indexing the camera/file and the element of the corresponding vector,
             respectively. E.g. rvec[i][j], refers to the rvec for the i-th camera,
             and the j-th element of it (out of total 3)
    """
#     extrinsic_files = _load_content_lines(_lst_files)
    rvec, tvec = [], []
    for _file in _lst_files:
        xmldoc = minidom.parse(_file)
        rvec.append([float(number)
                     for number in xmldoc.getElementsByTagName('rvec')[0].childNodes[0].nodeValue.strip().split()])
        tvec.append([float(number)
                     for number in xmldoc.getElementsByTagName('tvec')[0].childNodes[0].nodeValue.strip().split()])
    return rvec, tvec


def load_all_intrinsics(_lst_files):

    _cameraMatrices, _distCoeffs = [], []
    for _file in _lst_files:
        _cameraMatrices.append(load_opencv_xml(_file, 'camera_matrix'))
        _distCoeffs.append(load_opencv_xml(_file, 'distortion_coefficients'))
    return _cameraMatrices, _distCoeffs



# Calibration = namedtuple('Calibration', ['K', 'R', 'T', 'dist', 'view_id'])
def load_calibrations(root_path):

    intrinsic_path_format = "calibrations/intrinsic_zero/intr_{}.xml"
    extrinsic_path_format = "calibrations/extrinsic/extr_{}.xml"

    camera_id_to_name = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]

    intrinsic_pathes = [str(root_path / intrinsic_path_format.format(camera)) for camera in camera_id_to_name]
    extrinsic_pathes = [str(root_path / extrinsic_path_format.format(camera)) for camera in camera_id_to_name]

    rotationxyz, T = load_all_extrinsics(extrinsic_pathes)
    K, dist = load_all_intrinsics(intrinsic_pathes)
    
    calib_multi = list()
    for view_id in range(len(intrinsic_pathes)):
#         R = Rotation.from_euler('xyz', rotationxyz[view_id], degrees=False).as_matrix()
        R, _ = cv2.Rodrigues(np.array(rotationxyz[view_id]))

        # dist=dist[view_id]
        calib_multi.append(Calibration(K=K[view_id], R=R, T=np.array(T[view_id])[..., np.newaxis], view_id=view_id))

    return calib_multi


# Annotation = namedtuple('Annotation', ['xc', 'yc', 'w', 'h', 'feet', 'head', 'height', 'id', 'frame', 'view'])
def read_json(filename, calib_multi):
    """
    Decodes a JSON file & returns its content.
    Raises:
        FileNotFoundError: file not found
        ValueError: failed to decode the JSON file
        TypeError: the type of decoded content differs from the expected (list of dictionaries)
    :param filename: [str] name of the JSON file
    :return: [list] list of the annotations
    """
    if not os.path.exists(filename):
        raise FileNotFoundError("File %s not found." % filename)
    try:
        with open(filename, 'r') as _f:
            _data = json.load(_f)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode {filename}.")
    if not isinstance(_data, list):
        raise TypeError(f"Decoded content is {type(_data)}. Expected list.")
    if len(_data) > 0 and not isinstance(_data[0], dict):
        raise TypeError(f"Decoded content is {type(_data[0])}. Expected dict.")
        
    multi_view_gt = defaultdict(list)
    
    for person in _data:
        person_id = int(person["personID"])
        frame_id = int(Path(filename).stem)

        for bbox_v in person["views"]:
            if bbox_v["xmax"] == -1:
                continue
            view_id = int(bbox_v["viewNum"])
            xc = (bbox_v["xmax"] + bbox_v["xmin"]) / 2.0
            yc = (bbox_v["ymax"] + bbox_v["ymin"]) / 2.0
            w = bbox_v["xmax"] - bbox_v["xmin"]
            h = bbox_v["ymax"] - bbox_v["ymin"]

            bbox = Bbox(xc=xc, yc=yc, w=w, h=h)
            
            #Compute estimation for position of head and feet
            bbox_bottom_center = np.array([[xc], [h / 2.0 + yc], [1]])
            bbox_top_center = np.array([[xc], [- h / 2.0 + yc], [1]])
            
            calib = calib_multi[view_id]
            K, R, T = calib.K, calib.R, calib.T
            
            #Compute feet and head position in image plane (feet[0]) and in 3d world (feet[1])
            feet_reproj, feet_world  = project_feet(bbox_bottom_center, K, R, T, K, R, T)
            head_reproj, head_world  = project_head(feet_world, bbox_top_center, K, R, T, K, R, T)
            
            height = np.linalg.norm(head_world[1]-feet_world[1])
            
            multi_view_gt[view_id].append(Annotations(bbox=bbox, head=head_reproj,  feet=feet_reproj, height=height, id=person_id, frame=frame_id, view=view_id))
            
    return multi_view_gt


def project_feet(center_bottom, K0, R0, T0, K1, R1, T1):
    
    feet_world = geometry.reproject_to_world_ground(center_bottom, K0, R0, T0)    
    feet_reproj = geometry.project_world_to_camera(feet_world, K1, R1, T1)
    
    return feet_reproj, feet_world
    
def project_head(feet_world, center_top, K0, R0, T0, K1, R1, T1, average_height=165):
    
    head_world = feet_world.copy()
    head_world[2] = average_height

    head_reproj = geometry.project_world_to_camera(head_world, K1, R1, T1)
    
    return head_reproj, head_world