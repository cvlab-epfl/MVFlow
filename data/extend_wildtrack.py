import argparse
import json
import os

import cv2
import numpy as np

from pathlib import Path
from tqdm import tqdm

from dataset import wildtrack
from dataset.utils import get_frame_from_video, Calibration



def read_json(filename):
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
     
    return _data


def get_gt(index, root_wildtrack):
    gt_path = root_wildtrack / "annotations_positions/" / "{:08d}.json".format(index)
    gt = read_json(gt_path)
    
    return gt


def load_calibrations(root_path):

    intrinsic_path_format = "calibrations/intrinsic_original/intr_{}.xml"
    extrinsic_path_format = "calibrations/extrinsic/extr_{}.xml"

    camera_id_to_name = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]

    intrinsic_pathes = [str(root_path / intrinsic_path_format.format(camera)) for camera in camera_id_to_name]
    extrinsic_pathes = [str(root_path / extrinsic_path_format.format(camera)) for camera in camera_id_to_name]

    rotationxyz, T = wildtrack.load_all_extrinsics(extrinsic_pathes)
    K, dist = wildtrack.load_all_intrinsics(intrinsic_pathes)
    
    calib_multi = list()
    dist_multi = list()
    for view_id in range(len(intrinsic_pathes)):
#         R = Rotation.from_euler('xyz', rotationxyz[view_id], degrees=False).as_matrix()
        R, _ = cv2.Rodrigues(np.array(rotationxyz[view_id]))

        # dist=dist[view_id]
        calib_multi.append(Calibration(K=K[view_id], R=R, T=np.array(T[view_id])[..., np.newaxis], view_id=view_id))
        dist_multi.append(dist[view_id])
        
    return calib_multi, dist_multi

def interpolate_annotations(index, root_wildtrack):
    
    assert index % 5 != 0
    
    pre_ind = index - (index % 5)
    post_ind = pre_ind + 5
    
    pre_gt = get_gt(pre_ind, root_wildtrack)
    post_gt = get_gt(post_ind, root_wildtrack)
    
    inter_weight = (index - pre_ind) / 5
    
    ann_interpolated = list()

    for ann in pre_gt:
        ann_after = [ann_after for ann_after in post_gt if ann_after["personID"] == ann["personID"]]
        if len(ann_after) > 0:
            assert len(ann_after) == 1
            
            ann_inter = dict()
            ann_inter["personID"] = ann["personID"]
            ann_inter["positionID"] = ann["positionID"]
            ann_inter["views"] = list()
            
            for view, view_after in zip(ann["views"], ann_after[0]["views"]):
                view_dict = dict()
                view_dict["viewNum"] = view["viewNum"]
                
                if view["xmax"] == -1 or view_after["xmax"] == -1:
                    view_dict["xmax"] = -1
                else:
                    view_dict["xmax"] = int(np.rint(view["xmax"]*(1-inter_weight) + view_after["xmax"]*inter_weight))
                
                if view["xmin"] == -1 or view_after["xmin"] == -1:
                    view_dict["xmin"] = -1
                else:
                    view_dict["xmin"] = int(np.rint(view["xmin"]*(1-inter_weight) + view_after["xmin"]*inter_weight))
                
                if view["ymax"] == -1 or view_after["ymax"] == -1:
                    view_dict["ymax"] = -1
                else:
                    view_dict["ymax"] = int(np.rint(view["ymax"]*(1-inter_weight) + view_after["ymax"]*inter_weight))
                
                if view["ymin"] == -1 or view_after["ymin"] == -1:
                    view_dict["ymin"] = -1
                else:
                    view_dict["ymin"] = int(np.rint(view["ymin"]*(1-inter_weight) + view_after["ymin"]*inter_weight))
                
                
                ann_inter["views"].append(view_dict)
            
            ann_interpolated.append(ann_inter)
           
    return ann_interpolated


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("video_folder", help='path to the folder containing the wildtrack original videos')
    args = parser.parse_args()

    #reading wildtrack scenesetconfig to get path to dataset
    with open("./data/SceneSetConfigs/wildtrack_train.json", 'r') as _f:
        wildtrack_scene_set_config = json.load(_f)
    root_wildtrack = Path(wildtrack_scene_set_config["data_root"])

    #checking that wildtrack path and path to video exist
    assert root_wildtrack.is_dir(), "data_root in './data/SceneSetConfigs/wildtrack_train.json' is not pointing to a directory"
    assert Path(args.video_folder).is_dir(), "argument video_folder is not an existing folder"

    calibs, dists = load_calibrations(root_wildtrack)

    #generating intermediate frame and groundtruth
    for i in tqdm(range(5,2000)):
        if i % 5 == 0:
            continue
        
        index = 16 + (i - 5)*6
        for cam_id in [1,2,3,4,5,6,7]:
            video_path = args.video_folder + f"/cam{str(cam_id)}.mp4"
            frame = get_frame_from_video(video_path, index)
            frame = cv2.undistort(frame, calibs[cam_id-1].K, dists[cam_id-1])
            
            extend_frame_path = root_wildtrack / "Image_extended_subsets/" / "C{:d}/{:08d}.png".format(cam_id, i)
            extend_frame_path.parents[0].mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(extend_frame_path), frame)
    
        extend_gt = interpolate_annotations(i, root_wildtrack)
        
        outfile = root_wildtrack / "annotations_extended_positions/" / "{:08d}.json".format(i)
        outfile.parents[0].mkdir(parents=True, exist_ok=True)
        
        with open(str(outfile), 'w') as outfile:
            json.dump(extend_gt, outfile)
    

    #Generate a new scene_set_config for wildtrack extended with higher fps information
    wildtrack_extended_scene_set_config = wildtrack_scene_set_config.copy()
    wildtrack_extended_scene_set_config["fps"] = 10
    with open("./data/SceneSetConfigs/wildtrack_extended_train.json", 'w') as outfile:
            json.dump(wildtrack_extended_scene_set_config, outfile)
