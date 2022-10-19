import numpy as np
import torch 

from sklearn.cluster import KMeans


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


def decode_heatmap(heatmap, kernel_size, use_nms, threshold=None, K=150, boundary_det_withd = 0.05):
    '''
    Convert a heatmap to a set of 2d coordinate coresponding to positive detection
    K define max number of point selected in image
    Use threshold to filter out point with score lower than threshold
    Set threshold to auto to use kmean instead
    '''   
    
    if use_nms:
        heatmap = _nms(heatmap, kernel_size)

    topk_scores, ys, xs = _topk(heatmap.squeeze(0), K=K)

    
    if threshold is not None:
        if threshold=="auto":
            kmeans = KMeans(n_clusters=2, random_state=0).fit(topk_scores.reshape(-1,1))

            #If both cluster center are close to each other it means there isn't any positive element
            if np.abs(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]) < boundary_det_withd:
                mask = np.zeros_like(kmeans.labels_) != 0
            else:
                mask = kmeans.labels_ == kmeans.cluster_centers_.argmax()
        else:
            mask = topk_scores > threshold
        
        xs = xs[mask]
        ys = ys[mask]
        topk_scores = topk_scores[mask]

    pred_point = np.vstack([xs, ys]).T.astype(int)
    
    return topk_scores, pred_point