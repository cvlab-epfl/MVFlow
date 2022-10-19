import cv2
import numpy as np
import torch

from scipy.ndimage.filters import gaussian_filter


def gaussian_density_heatmap(size, points, radius):
    
    points = np.array(points)
    
    hm = np.zeros(size)

    if points.shape[0] != 0:
        hm[points[:,1],points[:,0]] = 1
        hm = gaussian_filter(hm, radius)

    hm = torch.from_numpy(hm).to(torch.float32)

    return hm

def gausian_center_heatmap(size, points, radius):

    points = np.array(points)
    
    hm = np.zeros(size)

    for point in points:
        draw_umich_gaussian(hm, point, radius, k=1)

    hm = torch.from_numpy(hm).to(torch.float32)

    return hm


def constant_center_heatmap(size, points, radius, value=1, background="zero"):
    
    if background == "zero":
        hm = np.zeros(size)
    elif background == "one":
        hm = np.ones(size)

    for point in points:
        cv2.circle(hm, tuple(point), radius, value, thickness=cv2.FILLED)

    hm = torch.from_numpy(hm).to(torch.float32)

    return hm




def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    # import pdb; pdb.set_trace()
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
        
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    # import pdb; pdb.set_trace()
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap