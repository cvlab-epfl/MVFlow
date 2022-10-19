from collections import defaultdict, Counter

import motmetrics as mm
import numpy as np
import pandas as pd
import torch 

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from dataset.utils import generate_flow
from misc.log_utils import log
from misc.utils import dict_merge

def compute_detection_stats(input_data, output_data):
    hm_flow = output_data["flow_pred"]["reconstruction_overall_density"].detach().unsqueeze(0)
    hm_flow = hm_flow - hm_flow.min()
    roi_rescale = input_data["ROI_mask"]
    hm_flow = hm_flow * roi_rescale
    hm_flow = hm_flow.cpu()
    
    gt_points_resized_flow = rescale_keypoints(input_data["gt_points"][0], input_data["hm"].shape[-2:], hm_flow.shape[-2:])

    precision_flow, recall_flow, TP_flow, FP_flow, FN_flow, flow_count, pred_point_flow = compute_scores_from_hm(hm_flow, gt_points_resized_flow, 3, True)

    gt_count = input_data["hm"].sum().item()

    output = {
        "TP_flow" : TP_flow,
        "FP_flow" : FP_flow,
        "FN_flow" : FN_flow,
        "prec_flow":precision_flow,
        "rec_flow":recall_flow,
        "flow_count_det_ae" : abs(flow_count - gt_count),
            }
    
    return output, pred_point_flow


def compute_count_metrics(input_data, processed_results):
    
    gt_count = len(input_data["gt_points"])
    
    #Iterate over all the predicted count from our model
    count_result = {}
    for key, pred_count in processed_results.items():
        if key[:5] == "count":
            count_result[key+"_ae"] = abs(pred_count - gt_count)

    return count_result


def compute_count_mae(stat_meter):
    sum_dict = stat_meter.sum()

    mae_result = {}
    for key, pred_ae_sum in sum_dict.items():
        # log.debug(key)
        if key[:5] == "count":
            # log.debug(key[:-2]+"mae")
            mae_result[key[:-2]+"mae"] = pred_ae_sum / stat_meter[key].count

    return mae_result

def compute_detection_metrics(input_data, processed_results, dist_threshold):

    # Prec_center, Rec_center, TP_center, FP_center, FN_center = compute_metrics_nearest(input_data["gt_points"][0], processed_results["pred_point_center"], dist_threshold)
    Prec_flow, Rec_flow, TP_flow, FP_flow, FN_flow = compute_metrics_nearest(input_data["gt_points"], processed_results["pred_point_flow"], dist_threshold)

    output = {
        # "TP_center" : TP_center,
        # "FP_center" : FP_center,
        # "FN_center" : FN_center,
        "TP_flow" : TP_flow,
        "FP_flow" : FP_flow,
        "FN_flow" : FN_flow,
        # "Prec_center" : Prec_center,
        # "Rec_center" : Rec_center,
        "Prec_flow" : Prec_flow,
        "Rec_flow" : Rec_flow,
            }

    return output


def count_single_flow_stats(flow_gt_list, pred_flow):
    flow_count_per_channel = defaultdict(list)

    for (pos, flow_channel) in flow_gt_list:
        flow_count_per_channel[flow_channel].append(pred_flow[flow_channel][pos[1]][pos[0]].item())

    return flow_count_per_channel

def compute_flow_metrics(input_data, output_data, conf, is_flow=True):

    if is_flow:
        flow_count_per_channel = defaultdict(list)

        _, flow_0_1f_gt, _ = generate_flow(input_data["gt_points_0"][0], input_data["person_id_0"][0], input_data["gt_points_1"][0], input_data["person_id_1"][0], input_data["ROI_mask"], hm_radius=conf["data_conf"]["hm_radius"], generate_hm=False)
        _, flow_1_2f_gt, _ = generate_flow(input_data["gt_points_1"][0], input_data["person_id_1"][0], input_data["gt_points_2"][0], input_data["person_id_2"][0], input_data["ROI_mask"], hm_radius=conf["data_conf"]["hm_radius"], generate_hm=False)
        _, flow_1_0b_gt, _ = generate_flow(input_data["gt_points_1"][0], input_data["person_id_1"][0], input_data["gt_points_0"][0], input_data["person_id_0"][0], input_data["ROI_mask"], hm_radius=conf["data_conf"]["hm_radius"], generate_hm=False)
        _, flow_2_1b_gt, _ = generate_flow(input_data["gt_points_2"][0], input_data["person_id_2"][0], input_data["gt_points_1"][0], input_data["person_id_1"][0], input_data["ROI_mask"], hm_radius=conf["data_conf"]["hm_radius"], generate_hm=False)

        prev_flow_stats  = count_single_flow_stats(flow_0_1f_gt, output_data["flow_pred"]["flow_0_1f"][0])
        post_flow_stats  = count_single_flow_stats(flow_1_2f_gt, output_data["flow_pred"]["flow_1_2f"][0])
        prev_flow_inverse_stats  = count_single_flow_stats(flow_1_0b_gt, output_data["flow_pred"]["flow_1_0b"][0])
        post_flow_inverse_stats  = count_single_flow_stats(flow_2_1b_gt, output_data["flow_pred"]["flow_2_1b"][0])

        flow_count_per_channel = dict_merge(prev_flow_stats, post_flow_stats, prev_flow_inverse_stats, post_flow_inverse_stats, empty_dict=defaultdict(list))
    else:
        #TODO implement metric on offset
        flow_count_per_channel = defaultdict(int)

    return flow_count_per_channel


def compute_scores_from_hm(heat_map, points_gt, kernel_size, use_nms):
    scores, pred_point = decode_heatmap(heat_map, kernel_size, use_nms, threshold="auto")
    
    precision, recall, TP, FP, FN = compute_metrics_nearest(points_gt, pred_point, kernel_size)

    return precision, recall, TP, FP, FN, pred_point.shape[0], pred_point


def compute_recall_precision(stat_meter):
    sum_dict = stat_meter.sum()

    if (sum_dict["TP_flow"] + sum_dict["FP_flow"]) == 0:
        log.warning("Cannot compute precision flow TP and FP are 0")
        precision_flow = 0
    else:
        precision_flow = sum_dict["TP_flow"] / (sum_dict["TP_flow"] + sum_dict["FP_flow"])
    
    if (sum_dict["TP_flow"] + sum_dict["FN_flow"]) == 0:
        log.warning("Cannot compute recall flow TP and FN are 0")
        recall_flow = 0
    else:
        recall_flow = sum_dict["TP_flow"] / (sum_dict["TP_flow"] + sum_dict["FN_flow"])

    output = {
        # "precision_center" : precision_center,
        # "recall_center" : recall_center,
        "precision_flow" : precision_flow,
        "recall_flow" : recall_flow
        }

    return output


def _nms(heatmap, kernel):
    pad = (kernel - 1) // 2
    
    #normalize heatmap such that it has a min of 0
    heatmap_min = heatmap.min()
    heatmap = heatmap - heatmap_min
    
    hmax = torch.nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    
    #shift heatmap back to it's orginal mean afeter aplying nms
    filtered_heatmap = (heatmap * keep) + heatmap_min
    
    return filtered_heatmap



def _topk(scores, K=100):
    scores = scores.squeeze()
    height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(-1), K)

    topk_inds = (topk_inds % (height * width))
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()

    return topk_scores.numpy(), topk_ys.numpy(), topk_xs.numpy()


def decode_heatmap(heatmap, kernel_size, use_nms, threshold=None, K=150):
    '''
    Convert a heatmap to a set of 2d coordinate coresponding to positive detection
    K define max number of point selected in image
    Use threshold to filter out point with score lower than threshold
    Set threshold to auto to use kmean instead
    '''   
    
    if use_nms:
        heatmap = _nms(heatmap, 3)

    topk_scores, ys, xs = _topk(heatmap.squeeze(0), K=K)

    
    if threshold is not None:
        if threshold=="auto":
            kmeans = KMeans(n_clusters=2, random_state=0).fit(topk_scores.reshape(-1,1))
            dist_beetween_cluster = np.abs(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1])
            if dist_beetween_cluster < 0.05:
                mask = np.zeros_like(kmeans.labels_) == 0
            else:
                mask = kmeans.labels_ == kmeans.cluster_centers_.argmax()
        else:
            mask = topk_scores > threshold
        
        xs = xs[mask]
        ys = ys[mask]
        topk_scores = topk_scores[mask]


    pred_point = np.vstack([xs, ys]).T
    
    return topk_scores, pred_point


def rescale_keypoints(points, org_img_dim, out_img_dim):

    out_img_dim = np.array(out_img_dim)
    org_img_dim = np.array(org_img_dim)
    
    if np.all(org_img_dim == out_img_dim):
        return points
    
    resize_factor = out_img_dim / org_img_dim
    #swap x and y
    resize_factor = resize_factor[::-1]


    resized_points = points*resize_factor
    
    return resized_points


def match_point_to_gt(gt_points, pred_points, dist_threshold):
    dist_mat = cdist(gt_points, pred_points)
    
    #Max linking distance above threshold all distance are equally large
    mask = dist_mat < dist_threshold
    dist_mat = dist_mat*mask
    dist_mat = dist_mat + (1-mask)*100000

    
    gt_ind, pred_ind = linear_sum_assignment(dist_mat)
    match_dist = dist_mat[gt_ind, pred_ind]

    
    return gt_ind, pred_ind, match_dist
    

    
def compute_metrics(gt_points, pred_points, match_distance, dist_threshold=3):
    TP = (match_distance < dist_threshold).sum()
    FP = pred_points.shape[0] - TP
    FN = gt_points.shape[0] - TP
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    return precision, recall, TP, FP, FN


def compute_metrics_nearest(gt_points, pred_points, dist_threshold):
    TP = 0
    FP = 0
    FN = 0

    if pred_points.shape[0] == 0:
        return 0, 0, TP, FP, len(gt_points)

    if len(gt_points) == 0:
        return 0, 0, TP, len(pred_points), 0

    for gt_p in gt_points:
        if find_nearest_point_distance(gt_p, pred_points) < dist_threshold:
            TP = TP + 1
        else:
            FN = FN + 1

    for pred_p in pred_points:
        if find_nearest_point_distance(pred_p, gt_points) > dist_threshold:
            FP = FP + 1
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    return precision, recall, TP, FP, FN


def find_nearest_point_distance(point, point_list):
    
    point_list = np.asarray(point_list)
    dist_2 = np.sum((point_list - point)**2, axis=1)
    
    index_nearest = np.argmin(dist_2)
    dist = np.sqrt(dist_2[index_nearest])

    return dist


def compute_mot_metric(gt_df, pred_df, metric_threshold, nb_gt):

    if gt_df.size == 0:
        log.spam("Trying to compute tracking metric on an empty sequence (gt size is 0)")
        return None

    acc = mm.utils.compare_to_groundtruth(gt_df, pred_df, 'euc', distfields=['X', 'Y'], distth=metric_threshold)
    
    #library doesn't implement moda computation, compute it manually form accumulator
    moda = compute_moda(acc, nb_gt)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')

    # log.debug(summary)
    metrics = dict(zip(summary.keys(), summary.values[0]))
    metrics["moda"] = moda
    
    return metrics

def compute_moda(accumulator, nb_gt):
    accdf = accumulator.mot_events

    frame_stats = dict()
    for index, row in accdf.iterrows():       
        frame_id = index[0]

        if frame_id not in frame_stats:
            frame_stats[frame_id] = list()
        
        frame_stats[frame_id].append(row[0])

    missandFP = 0
    total = 0
    for frame_id, stat_list in frame_stats.items():
        count = Counter(stat_list)
        
        missandFP = missandFP + count["MISS"] + count["FP"]
    
    if sum(nb_gt) != 0:
        moda = 1 - (missandFP) / sum(nb_gt)
    else:
        moda = 0

    return moda
    

def compute_mot_metric_from_det(gt, det, metric_threshold):
    """
    Wrapper function to compute MOT metric from detection
    convert detection to dataframe then call mm library
    """

    groundtruth_det_df = make_dataframe_from_det(gt)
    pred_det_df = make_dataframe_from_det(det)

    #list containing the number of gt detectino in every frame
    nb_gt = [len(x) for x in gt]

    metrics = compute_mot_metric(groundtruth_det_df, pred_det_df,metric_threshold, nb_gt)

    #Without any identity label, MOTA and MODA should be indentical
    assert metrics["moda"] == metrics["mota"]

    return metrics


def make_dataframe_from_det(det_list):
    det_as_list = list()
    for frame_id, frame_det in enumerate(det_list):
        det_as_list.extend([{'FrameId':frame_id, 'Id':-1, 'X':int(det[0]), 'Y':int(det[1])} for det in frame_det])

    det_as_df = pd.DataFrame(det_as_list)

    if  det_as_df.empty:
        det_as_df = pd.DataFrame(columns =['FrameId','Id','X','Y'])

    det_as_df = det_as_df.set_index(['FrameId', 'Id'])

    return det_as_df
