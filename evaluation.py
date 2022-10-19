import argparse
import sys
import time
import warnings
from collections import defaultdict
from multiprocessing import Queue

import numpy as np
import torch

from configs.arguments import get_config_dict
from dataset import factory as data_factory
from loss import factory as loss_factory
from misc import detection, geometry, metric, visualization
from misc.log_utils import DictMeter, batch_logging, dict_to_string, log
from misc.metric import compute_mot_metric_from_det
from misc.utils import dict_merge, listdict_to_dictlist
from model import factory as model_factory
from tracker import tracker

warnings.filterwarnings("ignore", category=UserWarning)

class Evaluator():
    def __init__(self, val_loaders, model, criterion, epoch, conf, use_tracker=False, nb_tracker=2):
        super(Evaluator, self).__init__()

        self.val_loaders = val_loaders
        self.model = model
        self.criterion = criterion

        self.epoch = epoch
        self.conf = conf

        self.total_nb_frames = sum([len(val_loader) for val_loader in self.val_loaders])

        self.loss_to_print = conf["training"]["loss_to_print"]
        self.metric_to_print = conf["training"]["metric_to_print"]

        #visualization and metric parameters
        self.detection_to_evaluate = conf["training"]["detection_to_evaluate"]
        self.motion_to_evaluate = conf["training"]["motion_to_evaluate"]
        self.use_nms = True
        self.nms_kernel_size = 3
        self.metric_threshold = 2.5

        self.use_tracker = use_tracker
        self.nb_tracker = nb_tracker

        self.tracker_processes = []
        self.best_valid_result = None

    def reset(self):
        
        #Before resetting make sure trackers completed their task
        self.waiting_for_tracker_to_finish()
        if self.use_tracker:
            #Initialize queues
            self.process_queues = [Queue() for x in range(self.nb_tracker)]
            self.result_queues = [Queue() for x in range(self.nb_tracker)]
            
            #initialize tracker
            self.tracker_processes = [
                tracker.MuSSPTracker(self.conf, self.epoch, self.process_queues[0], False, result_queue=self.result_queues[0]), 
                tracker.MuSSPTracker(self.conf, self.epoch, self.process_queues[1], True, result_queue=self.result_queues[1]), 
                ]

            assert self.nb_tracker == len(self.tracker_processes), "Nb tracker should be equal to the number of tracker_processes"
            #Start tracker
            [track_proc.start() for track_proc in self.tracker_processes]
            self.previous_put_index = 0
        else:
            self.process_queues = []

        self.stats_meter = DictMeter()
        self.flow_stats = list()
        self.epoch_result_dicts = list()
        self.model.eval()

        self.is_best = False

    def waiting_for_tracker_to_finish(self):
        if len(self.tracker_processes) != 0:
            #Check previous tracker have finished and print metrics
            log.debug("Waiting for tracker to finish")
            #if tracker are still running we wait for them to finish before starting next validation
            [track_proc.join() for track_proc in self.tracker_processes]
            log.debug(f"Tracking for epoch {self.epoch-1} completed")

            #Display results from tracker processes
            for res_queue in self.result_queues:
                metrics_dict = res_queue.get(block=True)
                if metrics_dict == -1:
                    log.error(f"Something went wrong in the tracker, skipping to next tracker")
                    continue

                tracker_type = metrics_dict.pop("tracker_type")
                log.info(f"{'':-^150}")
                log.info(f"Tracking result with {tracker_type}: \n"+dict_to_string(metrics_dict))

    def run(self, epoch):
        
        self.epoch = epoch

        #Wait for tracker and reset variable
        self.reset()

        end = time.time()
        for s, val_loader in enumerate(self.val_loaders):
            
            #reset some shared variable when switching scene
            step_dict = None

            for f, input_data in enumerate(val_loader):
                #global index of current frame
                i = sum([len(self.val_loaders[x]) for x in range(s)]) + f

                input_data = input_data.to(self.conf["device"])
                
                data_time = time.time() - end

                with torch.no_grad():
                    output_data = self.model(input_data)
                    
                    end2 = time.time()
                    if ("eval_metric" in self.conf["main"] and self.conf["main"]["eval_metric"]) or "eval_metric" not in self.conf["main"]:
                        criterion_output = self.criterion(input_data, output_data)
                    else:
                        criterion_output = {"stats":{}}

                    criterion_time = time.time() - end2

                    #put all the output in cpu to free gpu memory for the remaining of validation
                    output_data = output_data.to("cpu")
                    input_data = input_data.to("cpu")
                    
                    # Extract detected point
                    processed_results, output_data = self.post_process_heatmap(input_data, output_data)
                    #Compute detection and count metric if groundtruth is available
                    metric_stats = self.compute_metric(input_data, output_data, processed_results)
                    #Store data needed for tracking and visualization
                    self.store_step_dict(input_data, output_data, processed_results, metric_stats)
                
                batch_time = time.time() - end

                epoch_stats_dict = {**criterion_output["stats"], **metric_stats, **output_data["time_stats"], "batch_time":batch_time, "data_time":data_time, "criterion_time":criterion_time, "optim_time":0}
                self.stats_meter.update(epoch_stats_dict)
                
                if i % self.conf["main"]["print_frequency"] == 0 or i == (self.total_nb_frames - 1):
                    batch_logging(self.epoch, i, self.total_nb_frames, self.stats_meter, loss_to_print=self.loss_to_print, metric_to_print=self.metric_to_print, validation=True)

                end = time.time()
                #When we have accumulated max_tracklet_lenght step dict or reach the en dof dataset we push the step dict to the tracker process
                if (i % (self.conf["main"]["max_tracklet_lenght"]) == (self.conf["main"]["max_tracklet_lenght"] - 1)) or i == (self.total_nb_frames - 1):
                    for queue in self.process_queues:
                        try:
                            queue.put(self.epoch_result_dicts[self.previous_put_index:i+1])
                        except:
                            log.error("Couldn't pass the validation step_dicts to the tracker process")
                    self.previous_put_index = i+1

                del input_data
        
        #convert the list of result to a dict
        self.combine_step_dict()
        self.compute_epoch_metrics()
        self.generate_visualization()

        stats = {**self.stats_meter.avg()}

        del self.epoch_result_dicts

        #Signal to the tracker process that validation is over 
        for queue in self.process_queues:
            queue.put(-1)

        #Using AUC for pose estimation to compare model
        if self.best_valid_result is None or ((stats["moda_rec_1o"]) > self.best_valid_result):
            self.best_valid_result = (stats["moda_rec_1o"])
            self.is_best = True
        
        return {"stats":stats}



    def post_process_heatmap(self, input_data, output_data):

        #post process detection heatmap from self.detection_to_evaluate list
        processed_results = dict()

        #Set prediction outside of ROI to zero
        for det_k in self.detection_to_evaluate:
            if det_k.split("_")[0] != "framepred":
                output_data["flow_pred"][det_k] = output_data["flow_pred"][det_k] * input_data["ROI_mask"]
        for motion_k in self.motion_to_evaluate:
            output_data["flow_pred"][motion_k] = output_data["flow_pred"][motion_k]  * input_data["ROI_mask"]

        for det_k in self.detection_to_evaluate:
            scores_flow, pred_point_flow = detection.decode_heatmap(output_data["flow_pred"][det_k], self.nms_kernel_size, self.use_nms, threshold="auto")

            processed_results[det_k+"_points"] = pred_point_flow
            processed_results[det_k+"_scores"] = scores_flow

        return processed_results, output_data

    
    def compute_metric(self, input_data, output_data, processed_results):
        metrics_dict = dict()
        
        #For all the detection we compute MODA metric
        for det_k in self.detection_to_evaluate:
            if det_k.split("_")[0] == "framepred":
                gt_points = input_data["gt_points_image_"+det_k.split("_")[1][0]]
            else:
                gt_points = input_data["gt_points_"+det_k.split("_")[1][0]]

            metric_k = compute_mot_metric_from_det(gt_points, [processed_results[det_k+"_points"]], self.metric_threshold)
            metrics_dict.update({k+"_"+det_k: v for k,v in metric_k.items()})
            
        #We compute additional metric on motion prediction (whether it is flow or offset)
        self.flow_stats.append(metric.compute_flow_metrics(input_data, output_data, self.conf))

        metric_stats = {**metrics_dict}

        return metric_stats

    def store_step_dict(self, input_data, output_data, processed_results, metric_stats):
        """
        Store combination of input and prediction to generate tracker, metrics, and visualiztion
        We assume batchsize is 1 and only take the first element of the batch and the first view
        """
        step_dict = {}

        #Adding detection
        for det_k in self.detection_to_evaluate:
             step_dict[det_k] = output_data["flow_pred"][det_k][0,0]
             step_dict[det_k+"_points"] = processed_results[det_k+"_points"]
             step_dict[det_k+"_scores"] = processed_results[det_k+"_scores"]

        #Adding frame and gt to step dict
        for frame_id in range(self.conf["data_conf"]["nb_frames"]):
            step_dict[f"gt_points_{frame_id}"] = input_data[f"gt_points_{frame_id}"][0].astype(int)
            step_dict[f"person_id_{frame_id}"] = input_data[f"person_id_{frame_id}"][0]
            step_dict[f"hm_{frame_id}"] = input_data[f"hm_{frame_id}"][0]
            step_dict[f"frame_{frame_id}"] = visualization.inverse_img_norm(geometry.project_to_ground_plane_pytorch(input_data[f"frame_{frame_id}"][:,0:1], input_data[f"homography"][:,0:1], self.conf["data_conf"]["homography_input_size"], self.conf["data_conf"]["homography_output_size"], [x*4 for x in self.conf["data_conf"]["homography_output_size"]]))
            step_dict[f"frame_image_{frame_id}"] = visualization.inverse_img_norm(input_data[f"frame_{frame_id}"][:,0])
            step_dict[f"frame_{frame_id}_true_id"] = input_data[f"frame_{frame_id}_true_id"][0]

        for motion_k in self.motion_to_evaluate:
            step_dict[motion_k] = output_data["flow_pred"][motion_k][0]
        
        step_dict["metric_stats"] = metric_stats
        step_dict["roi"] = input_data["ROI_mask"][0]
        step_dict["scene_id"] = input_data["scene_id"][0]
        
        step_dict["mask_boundary"] = input_data["ROI_boundary_mask"][0]
        step_dict["homography"] = input_data["homography"][0]

        self.epoch_result_dicts.append(step_dict)

    def combine_step_dict(self):
        #flatten epoch_result_dict
        #processed flow start separetely since it a list of defaultdict
        self.epoch_result_dicts = listdict_to_dictlist(self.epoch_result_dicts)
        # self.epoch_result_dicts["flow_stats"] = merged_flow_stat

    def compute_epoch_metrics(self):
        merged_flow_stat  = dict_merge(*[curr_flow_stat for curr_flow_stat in self.flow_stats if len(curr_flow_stat) > 0], empty_dict=defaultdict(list))
        #For flow model we display flow statistic for the 9 channel of the flow
        for k, v in merged_flow_stat.items():
            if len(v) != 0:
                positive = len([prob for prob in v if prob > 0.5])
                log.info(f"flow stats channel {k} : {positive}/{len(v)} = {positive/len(v)} average prob {np.mean(v)}")

    def generate_visualization(self):
        if not(self.conf["training"]["eval_visual"]):
            return None

        visualizations = {}

        self.detection_to_evaluate 
        self.motion_to_evaluate

        for det in self.detection_to_evaluate:
            visualizations[det] = visualization.visualize_hm_det(self.epoch_result_dicts, det, self.conf)

        visualization.save_visualization_as_video(self.conf["training"]["ROOT_PATH"], visualizations, self.conf["main"]["name"], self.epoch, out_type="mp4")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    
    ####### Configuration #######
    parser.add_argument("checkpoint_path", help='path to the checkpoint to evaluate')
    parser.add_argument("-vid", '--video_path', dest="video_path", default="", help="Path to video to use for evaluation")
    parser.add_argument("-n", '--name', dest="name", default="", help="eval name (placehodler")
    parser.add_argument("-vids", '--video_sequence', dest="video_sequence", type=float, nargs='+', default=(0, 1), help="interval of the video to use in the evaluation by default full video interval (0,1)")
    parser.add_argument("-dev", "--device", dest="device", help="select device to use either cpu or cuda", default="cuda")
    parser.add_argument("-bs", '--batch_size', dest="batch_size", type=int, default=1,  help="The size of the batches")
    parser.add_argument("-vis", '--eval_visual', dest="eval_visual", action='store_true', default=False, help="Create video visualization from evaluation outputs")
    parser.add_argument("-dmet", '--disable_metric', dest="disable_metric", action='store_true', default=False, help="Avoid computing metric during evaluation")
    parser.add_argument('-tr', "--train_eval", dest="train_eval", action='store_true', default=False, help="Run evaluation on the training set")
    parser.add_argument("-splt", "--split_proportion", dest="split_proportion", type=float, default=-1, help="Train val split proportion the first split_proportion percent of the frames are used for training, the rest for validation")
    parser.add_argument("-dset", "--dataset", dest="dataset", default=None, nargs='*', choices=["PETS", "PETSeval",  "Parkinglot", "wild", "pomswa", "pomswatrain", "pomswatrain2", "pomrayeval3", "mot20train1", "mot20train2", "mot20train3", "mot20train5", "mot20test4", "mot20test6", "mot20test7", "mot20test8"], help='Dataset to use for Training')
    parser.add_argument("-mtl", "--max_tracklet_lenght", dest="max_tracklet_lenght", help="Number of element processed between print", default=None)
    parser.add_argument("-dtrack", '--disable_tracker', dest="disable_tracker", action="store_false", default=True, help="if flag is used it disable the use of tracker during evaluation")


    args = parser.parse_args()

    checkpoint_dict = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)

    #remove checkpint path from arg list
    del sys.argv[1]

    config = get_config_dict(checkpoint_dict["conf"])
    log.debug("loaded conf: " + dict_to_string(config))
    
    if args.max_tracklet_lenght is not None:
        config["main"]["max_tracklet_lenght"] = int(args.max_tracklet_lenght)
    
    if not (args.split_proportion == -1):
        config["data_conf"]["split_proportion"] = args.split_proportion

    if args.dataset is not None:
        config["data_conf"]["dataset"] = []
        config["data_conf"]["eval-dataset"] = args.dataset

    config["data_conf"]["batch_size"] = args.batch_size
    config["data_conf"]["shuffle_train"] = False
    config["data_conf"]["video_sequence"] = args.video_sequence
    config["training"]["eval_visual"] = args.eval_visual
    config["main"]["eval_metric"] = not(args.disable_metric)

    config["main"]["print_frequency"] = 100
    ##################
    ### Initialization
    ##################
    config["device"] = torch.device('cuda' if torch.cuda.is_available() and args.device == "cuda" else 'cpu') 
    log.info(f"Device: {config['device']}")

    end = time.time()
    log.info("Initializing model ...")
    
    model = model_factory.pipelineFactory(config["data_conf"])
    model.load_state_dict(checkpoint_dict["state_dict"])
    model.to(config["device"])

    log.info(f"Model initialized in {time.time() - end} s")

    
    end = time.time()
    log.info("Loading Data ...")
    
    if args.video_path:
        config["data_conf"]["dataset"] = "video"
        config["data_conf"]["video_path"] = args.video_path
        config["data_conf"]["split_proportion"] = 0
        config["main"]["eval_metric"] = False

    train_dataloader, val_dataloader = data_factory.get_dataloader(config["data_conf"])

    if args.train_eval:
        dataloader = train_dataloader
        
    else:
        dataloader = val_dataloader

    log.info(f"Data loaded in {time.time() - end} s")

    criterion = loss_factory.get_loss(config["data_conf"], config["loss_conf"])

    ##############
    ### Evaluation
    ##############

    end = time.time()
    log.info(f"Beginning validation")
    evaluator = Evaluator(dataloader, model, criterion, checkpoint_dict["epoch"], config, use_tracker=True)

    valid_results = evaluator.run(checkpoint_dict["epoch"])
    
    evaluator.waiting_for_tracker_to_finish()


    log.info(f"Validation completed in {time.time() - end}s")