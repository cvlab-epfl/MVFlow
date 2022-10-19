from email.policy import default

import mot3d
import mot3d.weight_functions as wf
import mot3d.distance_functions as df
import motmetrics as mm
import pandas as pd
import numpy as np
import torch

from mot3d.utils import trajectory
from multiprocessing import Process
from pathlib import Path

from misc.log_utils import log
from misc.metric import compute_mot_metric
from misc.utils import dict_mean, flatten
from misc.visualization import save_visualization_as_video
from tracker.utils import Track, make_dataframe_from_tracks, get_nb_det_per_frame_from_tracks, suppress_stdout
from tracker.visualization import make_tracking_vis

class BaseTracker(Process):
    def __init__(self, conf, epoch, data_queue, result_queue=None, generate_viz=True):
        super(BaseTracker, self).__init__(daemon=True)
        self.conf = conf
        self.epoch = epoch

        self.data_queue = data_queue
        self.result_queue = result_queue

        self.generate_viz = generate_viz
        self.metric_threshold = 2.5

        self.time_id = 1
        self.next_time_id = 2
        self.flow_type = "flow_1_2f"
        self.flow_type_inv = "flow_2_1b"

        self.det_type = "rec_1o"

        self.groundplane_images = list()
        self.rois = list()
        self.gt_points = list()

        self.tracker_type = None

        self.all_tracks = list()
        self.all_scene_ids = list()
        self.all_homography = list()
        self.all_seq_starting_frame = list()

        self.prev_scene_id = -1
        self.current_frame_id = 0

        self.metrics = list()

        self.reset()

    def reset(self):
        self.tracklet_list = list()
        self.groundtruth_tracks = dict()
        self.pred_tracks = list()
    
    def run(self):
        log.spam(f"Starting tracker process {self.tracker_type} using {self.det_type}_points and motion {self.flow_type}")

        metrics = self.start_tracker_loop()

        if self.generate_viz:
            self.generate_visualization()

        if self.result_queue is not None:
            metrics["tracker_type"] = self.tracker_type
            self.result_queue.put(metrics)

    def start_tracker_loop(self):
        index_change = -1 

        #we loop until we get a -1 from the queue
        while True:
            if index_change == -1:
                try:
                    validation_sequence = self.data_queue.get(timeout=10)
                except:
                    continue

                if validation_sequence == -1:
                    #All the sequence from validation have been processed exiting
                    break
                
                if self.prev_scene_id == -1:
                    self.prev_scene_id = validation_sequence[0]["scene_id"]
                    self.curr_homography = validation_sequence[0]["homography"]
                    self.curr_starting_frame = self.current_frame_id

            log.spam(f"tracker {self.tracker_type} validation sequence lenght {len(validation_sequence)}")
            #check if there is a scene change in the next list
            index_change = min([i for i, el in enumerate(validation_sequence) if el["scene_id"] != self.prev_scene_id], default=-1)

            if index_change != -1:
                log.spam("New scene detected")
                validation_sequence_new_scene = validation_sequence[index_change:]
                validation_sequence = validation_sequence[:index_change]

            self.rois.extend([step_dict["roi"] for step_dict in validation_sequence])

            self.groundplane_images.extend([step_dict[f"frame_{self.time_id}"] for step_dict in validation_sequence])
            self.gt_points.extend([step_dict[f"gt_points_{self.time_id}"] for step_dict in validation_sequence])

            self.extend_groundtruth_tracks(validation_sequence)
            new_tracklets = self.generate_tracklets(validation_sequence)
            self.tracklet_list.append(new_tracklets)

            self.current_frame_id = self.current_frame_id + len(validation_sequence)

            if index_change != -1:
                log.spam("combining tracklets")
                #the scene has change we combine tracklet, compute metrics, and reset the tracker
                self.pred_tracks = self.combine_tracklets()
                self.all_tracks.append(self.pred_tracks)
                self.all_scene_ids.append(self.prev_scene_id)
                self.all_homography.append(self.curr_homography)
                self.all_seq_starting_frame.append(self.curr_starting_frame)
                curr_metric = self.compute_tracking_metric()
                if curr_metric is not None:
                    self.metrics.append(curr_metric)

                #reseting tracker
                self.reset()

                #setting up for next iteration and next scene 
                self.prev_scene_id = validation_sequence_new_scene[0]["scene_id"]
                self.curr_homography = validation_sequence_new_scene[0]["homography"]
                self.curr_starting_frame = self.current_frame_id
                validation_sequence = validation_sequence_new_scene
                continue

        self.data_queue.close()
        self.pred_tracks = self.combine_tracklets()
        self.all_tracks.append(self.pred_tracks)
        self.all_scene_ids.append(self.prev_scene_id)
        self.all_homography.append(self.curr_homography)
        self.all_seq_starting_frame.append(self.curr_starting_frame)
        curr_metric = self.compute_tracking_metric()
        if curr_metric is not None:
            self.metrics.append(curr_metric)

        metrics = dict_mean(self.metrics)

        return metrics

    def extend_groundtruth_tracks(self, step_dict_list):
        
        for i, step_dict in enumerate(step_dict_list):
            for person_id, point in zip(step_dict[f"person_id_{self.time_id}"], step_dict[f"gt_points_{self.time_id}"]):
                if person_id not in self.groundtruth_tracks:
                    self.groundtruth_tracks[person_id] = Track(person_id)

                self.groundtruth_tracks[person_id].add_detection(self.current_frame_id + i,point[0], point[1])

    def compute_tracking_metric(self):
        groundtruth_tracks_df = make_dataframe_from_tracks(self.groundtruth_tracks.values())
        pred_tracks_df = make_dataframe_from_tracks(self.pred_tracks)

        nb_gt = get_nb_det_per_frame_from_tracks(self.groundtruth_tracks.values()).values()
        
        return compute_mot_metric(groundtruth_tracks_df, pred_tracks_df, self.metric_threshold, nb_gt)

    def generate_visualization(self):
        pred_scale = 4

        visualization_result = make_tracking_vis(self.groundplane_images, flatten(self.all_tracks), f"{self.tracker_type} Tracking", self.rois, pred_scale, gt_points=self.gt_points)
        save_visualization_as_video(self.conf["training"]["ROOT_PATH"], {f"{self.tracker_type}":visualization_result}, self.conf["main"]["name"], self.epoch, out_type="mp4")

def get_dir_vec_from_flow(flow):
    flow_dir = np.argmax(flow[:9])

    if flow_dir == 0:
        return [-1,-1]
    if flow_dir == 1:
        return [0,-1]
    if flow_dir == 2:
        return [1,-1]
    if flow_dir == 3:
        return [-1,0]
    if flow_dir == 4:
        return [0,0]
    if flow_dir == 5:
        return [1,0]
    if flow_dir == 6:
        return [-1,1]
    if flow_dir == 7:
        return [0,1]
    if flow_dir == 8:
        return [1,1]


def flow_similarity_dist(d1_pos, d1_flow, d1_index, d2_pos, d2_index):
        
    flow_vec = get_dir_vec_from_flow(d1_flow)
    
    predict_pos = d1_pos + np.array(flow_vec)*(d2_index - d1_index)

    dist = wf.euclidean(predict_pos, d2_pos)

    return dist

def weight_distance_detections_2d_with_flow(d1, d2,
                                  sigma_jump=1, sigma_distance=2,
                                  sigma_color_histogram=0.3, sigma_box_size=0.3,  sigma_flow=3,
                                  max_distance=20,
                                  use_color_histogram=True, use_bbox=True, use_flow=True):
    weights = []

    dist = wf.euclidean(d1.position, d2.position)
    if dist>max_distance:
        return None
    weights.append( np.exp(-dist**2/sigma_distance**2) )    
    
    if use_color_histogram:
        chs = 1-df.color_histogram_similarity(d1.color_histogram, d2.color_histogram)
        weights.append( np.exp(-chs**2/sigma_color_histogram**2) )
    
    if use_bbox:
        bss = 1-df.bbox_size_similarity(d1.bbox, d2.bbox)
        weights.append( np.exp(-bss**2/sigma_box_size**2) )
    
    if use_flow:
        fls = flow_similarity_dist(d1.position, d1.flow, d1.index, d2.position, d2.index)
        weights.append(np.exp(-fls**2/sigma_flow**2) )
    
    jump = d1.diff_index(d2)
    return -np.exp(-(jump-1)*sigma_jump) * np.prod(weights)

class Detection2DFlow(mot3d.Detection2D):
    def __init__(self, index, position=None, confidence=0.5, id=None, 
                    view=None, color_histogram=None, bbox=None, flow=None):    
        super().__init__(index, position, confidence, id, view, color_histogram, bbox)
        self.flow = flow

class MuSSPTracker(BaseTracker):
    
    def __init__(self, conf, epoch, data_queue, use_flow, *args, **kwargs):
        super().__init__(conf, epoch, data_queue, *args, **kwargs)
        if use_flow:
            self.tracker_type = "mussp_flow"
        else:
            self.tracker_type = "mussp"

        self.dummy_color_histogram = [np.ones((128,))]*3
        self.dummy_bbox = [0,0,1,1]

        conf["sigma_jump"] = 1
        conf["sigma_distance"] = 10#3 
        conf["sigma_flow"] = 4

        self.weight_distance = lambda d1, d2: weight_distance_detections_2d_with_flow(d1, d2,
                                                                        sigma_jump=conf["sigma_jump"], sigma_distance=conf["sigma_distance"],
                                                                        sigma_color_histogram=0.3, sigma_box_size=0.3, sigma_flow=conf["sigma_flow"], #3.5 # 5
                                                                        max_distance=15,
                                                                        use_color_histogram=False, use_bbox=False, use_flow=use_flow)

        self.weight_confidence = lambda d: wf.weight_confidence_detections_2d(d, mul=1, bias=0)

        self.weight_distance_t = lambda t1, t2: wf.weight_distance_tracklets_2d(t1, t2, max_distance=None,
                                                                        sigma_color_histogram=0.3, sigma_motion=50, alpha=0.7,
                                                                        cutoff_motion=0.1, cutoff_appearance=0.1,
                                                                        use_color_histogram=False)
        
        self.weight_confidence_t = lambda t: wf.weight_confidence_tracklets_2d(t, mul=1, bias=0)

    def reset(self):
        BaseTracker.reset(self)

    def generate_tracklets(self, step_dict_list):
        
        if len(step_dict_list) == 0:
            return list()

        detections = list()
        for i, frame_dict in enumerate(step_dict_list):
            curr_detections = frame_dict[f"{self.det_type}_points"]
            curr_flow = frame_dict[f"{self.flow_type}"]

            detections.extend([Detection2DFlow(self.current_frame_id + i, det, color_histogram=self.dummy_color_histogram, bbox=self.dummy_bbox, flow=curr_flow[:,det[1],det[0]]) for det in curr_detections])

        with suppress_stdout():
            g = mot3d.build_graph(detections, weight_source_sink=0.1,
                                max_jump=4, verbose=False,
                                weight_confidence=self.weight_confidence,
                                weight_distance=self.weight_distance)

        if g is None:
            return list()

        with suppress_stdout():
            tracklets = mot3d.solve_graph(g, verbose=False, method='muSSP')  

        return tracklets

    def combine_tracklets(self):

        tracklets = flatten(self.tracklet_list)

        tracklets = mot3d.remove_short_trajectories(tracklets, th_length=2)
        detections_tracklets = [mot3d.DetectionTracklet2D(tracklet) for tracklet in tracklets]
        
        with suppress_stdout():
            g = mot3d.build_graph(detections_tracklets, weight_source_sink=0.1,
                                max_jump=20, verbose=False,
                                weight_confidence=self.weight_confidence_t,
                                weight_distance=self.weight_distance_t)    
        if g is None:
            log.warning(f"tracker {self.tracker_type} Combining tracklet there is not a single path between sink and source nodes! return empty tracking")
            return list() 
        
        with suppress_stdout():
            trajectories = mot3d.solve_graph(g, verbose=False, method='muSSP')

        tracks = dict()

        for i, track in enumerate(trajectories):
            tracks[i] = Track(i)
            
            for tracklet in track:
                tracklet = trajectory.interpolate_trajectory(tracklet.tracklet)
                for det in tracklet:
                    tracks[i].add_detection(det.index, det.position[0], det.position[1])

        return tracks.values()