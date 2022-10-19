import colorsys
from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import torch
from dataset.utils import get_frame_from_file, generate_motion_tuple
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torchvision import transforms

from misc.log_utils import log

def visualize_hm_det(epoch_result_dicts, det, conf):

    visualization_frames = list()
    f_id = det.split("_")[1][0]

    for frame_id in range(len(epoch_result_dicts[det])):

        metrics = epoch_result_dicts["metric_stats"]

        pred_scale = 4

        frame_flow = visualize_density_cv2(
            epoch_result_dicts[f"frame_{f_id}"][frame_id], 
            epoch_result_dicts[det][frame_id], 
            det, 
            points=epoch_result_dicts[det+"_points"][frame_id], 
            count=0, 
            count_det=len(epoch_result_dicts[det+"_points"][frame_id]), 
            prec=metrics[f"precision_{det}"][frame_id], 
            rec=metrics[f"recall_{det}"][frame_id], 
            gt_points=epoch_result_dicts[f"gt_points_{f_id}"][frame_id], 
            roi=epoch_result_dicts["roi"][frame_id],
            pred_scale=pred_scale
            )

        visualization_frames.append(frame_flow)

    return visualization_frames


def visualize_motion(epoch_result_dicts, motion, conf):
    visualization_frames = list()
    start_id = motion.split("_")[1][0]
    end_id = motion.split("_")[2][0]

    motion_direction = motion.split("_")[-1][-1]
    det = f"det_{start_id}{motion_direction}"
    rec = f"rec_{motion.split('_')[2]}"

    log.spam(f"Generating visulization for motion {motion} base on {det} and producing {rec}")

    for frame_id in range(len(epoch_result_dicts[motion])):

        metrics = epoch_result_dicts["metric_stats"]

        flow_pairs =  generate_motion_tuple(
            epoch_result_dicts[f"gt_points_{start_id}"][frame_id],
            epoch_result_dicts[f"person_id_{start_id}"][frame_id],
            epoch_result_dicts[f"gt_points_{end_id}"][frame_id],
            epoch_result_dicts[f"person_id_{end_id}"][frame_id]
            ) 

        frame_flow = generate_flow_map_with_arrow(
        epoch_result_dicts[f"frame_{start_id}"][frame_id], 
        epoch_result_dicts[det][frame_id], 
        epoch_result_dicts[rec+"_points"][frame_id], 
        epoch_result_dicts[motion][frame_id],
        flow_pairs,
        motion,
        legend_det=f"T{start_id}", 
        legend_rec=f"T{end_id}")

        visualization_frames.append(frame_flow)

    return visualization_frames



def visualize_bounding_box(bbox_list, frame_pathes):
    bb_frames = list()
    
    for i, frame_path in enumerate(frame_pathes):
        frame = get_frame_from_file(Path(frame_path))
        
        for bb in [bb for bb in bbox_list if bb[0] == i+1]:
            left = int(float(bb[2]))
            top = int(float(bb[3]))
            width = int(float(bb[4]))
            height = int(float(bb[5]))
            frame = cv2.rectangle(frame, (left,top), (left+width, top+height), (255,0,0), 2)
    
        bb_frames.append(frame)

    return bb_frames


def visualize_multi_hm(epoch_result_dicts, conf, type):

    
    visualization_frames = list()
    for frame_id in range(len(epoch_result_dicts["frame"])):

        results = epoch_result_dicts["processed_results"]
        metrics = epoch_result_dicts["metric_stats"]

        frame_center = visualize_density_cv2(
            epoch_result_dicts["frame_groundplane"][frame_id], 
            epoch_result_dicts[f"hm_{type}_ground_solo"][frame_id], 
            f"{type} Det. ground", 
            points=results[f"pred_point_{type}_ground"][frame_id], 
            gt_points=epoch_result_dicts["gt_points"][frame_id], 
            roi=epoch_result_dicts["roi"][frame_id]
            )
        
        frame_flow = visualize_density_cv2(
            epoch_result_dicts["frame_headplane"][frame_id], 
            epoch_result_dicts[f"hm_{type}_head_solo"][frame_id], 
            f"{type} Det. head", 
            points=results[f"pred_point_{type}_head"][frame_id],
            gt_points=epoch_result_dicts["gt_points"][frame_id], 
            roi=epoch_result_dicts["roi"][frame_id]
            )

        visualization_frames.append(combine_two_frame_cv2(frame_center, frame_flow))

    return visualization_frames

def visualize_track(epoch_result_dicts, conf, left="center", right="flow"):

    pred_scale = 4

    frames_track_left = make_tracking_vis(epoch_result_dicts["frame_groundplane"], epoch_result_dicts["tracking"][left], f"{left} Tracking", epoch_result_dicts["roi"], pred_scale)
    frames_track_right = make_tracking_vis(epoch_result_dicts["frame_groundplane"], epoch_result_dicts["tracking"][right], f"{right} Tracking", epoch_result_dicts["roi"], pred_scale)
        
    visualization_tracking_frames = list()

    for frame_id in range(len(frames_track_left)):
        visualization_tracking_frames.append(combine_two_frame_cv2(frames_track_left[frame_id], frames_track_right[frame_id]))

    return visualization_tracking_frames


def combine_two_frame_cv2(frame1, frame2):
    
    frame1 = cv2.copyMakeBorder(frame1, 0, 0, 0, 8, cv2.BORDER_CONSTANT, value=[255,255,255])
    frame2 = cv2.copyMakeBorder(frame2, 0, 0, 8, 0, cv2.BORDER_CONSTANT, value=[255,255,255])

    combined_frame = np.concatenate((frame1, frame2), axis=1)
    
    return combined_frame


def visualize_density_cv2(input_img, density_pred, title, points=None, count=None, count_det=None, prec=None, rec=None, gt_points=None, roi=None, pred_scale=4):
    # log.debug(density_pred.shape)
    density_pred = np.array(density_pred * 255, dtype = np.uint8).squeeze() #(density_pred.unsqueeze(0).numpy() * 255).astype(int)
    # log.debug(density_pred.shape)

    input_img = np.array(input_img * 255, dtype = np.uint8)
    density_pred = cv2.normalize(density_pred, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # log.debug(density_pred.shape)

    if pred_scale != 1:
        # log.debug(tuple([c*pred_scale for c in density_pred.shape[:2]]))
        density_pred = cv2.resize(density_pred, tuple([c*pred_scale for c in density_pred.shape[:2]][::-1]))
        # log.debug(density_pred.shape)

    heatmap_img = cv2.applyColorMap(density_pred, cv2.COLORMAP_RAINBOW)

    if input_img.shape[:2] != heatmap_img.shape[:2]:
        input_img = cv2.resize(input_img, heatmap_img.shape[:2])

    out_image = cv2.addWeighted(input_img, 0.8, heatmap_img, 0.5, 0)
    
    if roi is not None:
        roi = np.array(roi.squeeze() * 255, dtype = np.uint8)
        roi = cv2.applyColorMap(roi, cv2.COLORMAP_BONE )
        if pred_scale != 1:
            roi = cv2.resize(roi, tuple([c*pred_scale for c in roi.shape[:2]][::-1]), interpolation=cv2.INTER_NEAREST)
        out_image = cv2.addWeighted(out_image, 0.8, roi, 0.4, 0)
        
    #Draw detected points
    if points is not None:
        for point in points:
            # log.debug(point)
            out_image = cv2.drawMarker(out_image, tuple(point*pred_scale), tuple([190,0,0]), cv2.MARKER_CROSS, markerSize=4, thickness=1) #MarkerType[, markerSize[, thickness[, line_type]]]]	)
    
        textSize, baseline = cv2.getTextSize("Pred points", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.drawMarker(out_image, (out_image.shape[1] - textSize[0] - 15, int(out_image.shape[0] - 2*baseline - 2 * textSize[1])), [190,0,0], cv2.MARKER_CROSS, markerSize=4, thickness=1) #MarkerType[, markerSize[, thickness[, line_type]]]]	)
        cv2.putText(out_image, "Pred points", (out_image.shape[1] - textSize[0] - 5, int(out_image.shape[0] - 2*baseline - 1.5*textSize[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,255], thickness=1)

        out_image = cv2.rectangle(out_image, (out_image.shape[1] - textSize[0] - 30, int(out_image.shape[0] - 2*baseline - 2 * textSize[1] -15)), (out_image.shape[1], int(out_image.shape[0])), [255,255,255], 1)
    
    #Draw groundtruth points
    if gt_points is not None:
        # log.debug(gt_points)
        # log.debug("#######")
        for point in gt_points:
            # log.debug(point)
            out_image = cv2.drawMarker(out_image, tuple(point*pred_scale), [0,190,0], cv2.MARKER_SQUARE, markerSize=4, thickness=1) #MarkerType[, markerSize[, thickness[, line_type]]]]	)
    
        cv2.drawMarker(out_image, (out_image.shape[1] - textSize[0] - 15, int(out_image.shape[0] - 1.5*baseline - textSize[1] / 2)), [0,190,0], cv2.MARKER_SQUARE, markerSize=4, thickness=1)
        cv2.putText(out_image, "Gt points", (out_image.shape[1] - textSize[0] - 5, int(out_image.shape[0] - 1.5*baseline)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,255], thickness=1)
    
    #Add margin and captions
    
    out_image = cv2.copyMakeBorder(out_image, 32, 32, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    #Title Bar
    textSize, baseline = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    out_image = cv2.putText(out_image, title, (int(out_image.shape[1] / 2 - textSize[0] / 2), 23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255,255,255], thickness=1)
    
    #Bottom text add gt and metric if availble
    if count is not None and count_det is not None:
        count = round(min(count, 999))
        count_det = round(min(count_det, 999))

        bottom = f"Count - Hm {count} - Det {count_det} "
    else:
        bottom = ""
        
    if gt_points is not None:
        bottom = bottom + f"Gt - {str(len(gt_points))} "

    if prec is not None:
        bottom = bottom + f"|| Prec - {prec:.2f} Rec - {rec:.2f}"
    
    textSize, baseline = cv2.getTextSize(bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    out_image = cv2.putText(out_image, bottom, (int(out_image.shape[1] / 2 - textSize[0] / 2), 561), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,255,255], thickness=1)
    
    
    return out_image

# adapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def generate_colors(N):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: tuple(round(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv))
    perm = np.random.permutation(N)
    colors = [colors[idx] for idx in perm]
    
    return colors


def make_tracking_vis(frames, full_tracks, title, roi, pred_scale):
    out_frames = list()
    
    colors = generate_colors(len(full_tracks))

    for i, frame in enumerate(frames):
        
        # frame_inv = inverse_img_norm(frame)
        partial_tracks = []
        for track_id, full_track in enumerate(full_tracks):
            partial_tracks.append((track_id, get_local_track(full_track, i)))
        
        if roi is not None:
            roi_curr = roi[i]
        else:
            roi_curr = None

        vis = make_frame_visu_track(frame, partial_tracks, colors, title=title, roi=roi_curr, pred_scale=pred_scale)
        
        out_frames.append(vis)
        
    return out_frames


def get_local_track(full_track, timestamp, local_size=10):
    
    if timestamp < full_track[0][0] or timestamp > full_track[-1][0]:
        return []
    else:
        track_id = timestamp - full_track[0][0]
        return [track_point[1] for track_point in full_track[max(0,track_id-local_size):track_id]]           


def make_frame_visu_track(input_img, partial_tracks, colors, title, roi=None, pred_scale=4):
    
    out_image = np.ascontiguousarray(np.array(input_img * 255, dtype = np.uint8))
    
    
    if roi is not None:
        roi = np.array(roi.squeeze() * 255, dtype = np.uint8)
        roi = cv2.applyColorMap(roi, cv2.COLORMAP_BONE )
        if pred_scale != 1:
            roi = cv2.resize(roi, tuple([c*pred_scale for c in roi.shape[:2][::-1]]), interpolation=cv2.INTER_NEAREST)
        out_image = cv2.addWeighted(out_image, 0.8, roi, 0.2, 0)
    
    for track_id, partial_track in partial_tracks:
        partial_track = np.array(partial_track, np.int32)*pred_scale
        
        if partial_track.shape[0] != 0:
            cv2.drawMarker(out_image, (partial_track[-1,0], partial_track[-1,1]), colors[track_id], cv2.MARKER_STAR, markerSize=6, thickness=2) #MarkerType[, markerSize[, thickness[, line_type]]]]	)
            cv2.putText(out_image, str(track_id), (partial_track[-1,0] + 8, partial_track[-1,1]+3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[track_id], thickness=1)
            cv2.polylines(out_image, [partial_track.reshape((-1, 1, 2))], False, colors[track_id], thickness=2)

        
    out_image = cv2.copyMakeBorder(out_image, 32, 32, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    #Title Bar
    textSize, baseline = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    out_image = cv2.putText(out_image, title, (int(out_image.shape[1] / 2 - textSize[0] / 2), 23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255,255,255], thickness=1)
    
    return out_image
    

def inverse_img_norm(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    #Convert normalize pytorch tensor back to numpy image

    inv_normalize = transforms.Normalize(
        mean=[-mean[0]/0.229, -mean[1]/0.224, -mean[2]/0.225],
        std=[1/std[0], 1/std[1], 1/std[2]]
    )

    img_unorm = torch.clip(inv_normalize(img), 0, 1).squeeze().permute(1,2,0)

    return img_unorm.cpu().numpy()


def save_visualization_as_video(project_root, dict_visualization, model_id, epoch, out_type="avi"):
    
    for visu_type, frame_list in dict_visualization.items():
        file_name = project_root + "/results/{model_id}/{model_id}_epoch_{epoch}_{visu_type}".format(model_id=model_id, visu_type=visu_type, epoch=str(epoch))

        if out_type == "avi":
            save_video_avi(frame_list, file_name)
        elif out_type == "mp4":
            save_video_mp4(frame_list, file_name)

def save_video_mp4(frame_list, path, save_framerate=30):
    file_path = Path('{}.mp4'.format(path))
    file_path.parents[0].mkdir(parents=True, exist_ok=True)

    imageio.mimwrite(file_path, frame_list, fps=save_framerate)       

def save_video_avi(frames, path, save_framerate=30):
    video_h, video_w = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    file_path = Path('{}.avi'.format(path))
    
    #Check if parent dir exist otherwise make it
    file_path.parents[0].mkdir(parents=True, exist_ok=True)
    
    out = cv2.VideoWriter(str(file_path), fourcc, save_framerate, (video_w, video_h))
    
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()


def visualize_density(input_img, density_pred, roi=None):
      
    my_dpi=50
    fig = plt.figure(figsize=(float(input_img.shape[1])/my_dpi,float(input_img.shape[0])/my_dpi))
    canvas = FigureCanvasAgg(fig)
    
    ax=fig.add_subplot(111)

    # Remove whitespace from around the image
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    
    ax.imshow(input_img)
    ax.imshow(density_pred, alpha=0.4, cmap='rainbow')
    if roi is not None:
        ax.imshow(roi, alpha=0.2)
    
    canvas.draw()
    data = np.array(canvas.renderer.buffer_rgba())
    plt.close()
    
    return cv2.cvtColor(data, cv2.COLOR_RGBA2RGB)
    

def visualize_count(input_img, density_pred, grid_size=40, roi=None):

    my_dpi=50
    
    # Set up figure
    fig=plt.figure(figsize=(float(input_img.shape[1])/my_dpi,float(input_img.shape[0])/my_dpi))
    canvas = FigureCanvasAgg(fig)
    
    ax=fig.add_subplot(111)

    # Remove whitespace from around the image
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    # Set the gridding interval: here we use the major tick interval
  
    locx = plticker.MultipleLocator(base=grid_size)#plticker.LinearLocator(numticks=input_img.shape[1]//grid_size)
    locy = plticker.MultipleLocator(base=grid_size)#plticker.LinearLocator(numticks=input_img.shape[0]//grid_size)
    
    ax.xaxis.set_major_locator(locx)
    ax.yaxis.set_major_locator(locy)

    # Add the grid
    ax.grid(which='major', axis='both', linestyle='-', linewidth=1, color="black")
    
    # Add the image
    ax.imshow(input_img)
    
    if roi is not None:
        ax.imshow(roi, alpha=0.2)

    # Find number of gridsquares in x and y direction
    nx=abs(int(float(ax.get_xlim()[1]-ax.get_xlim()[0])/float(grid_size)))
    ny=abs(int(float(ax.get_ylim()[1]-ax.get_ylim()[0])/float(grid_size)))

    # Add some labels to the gridsquares
    for j in range(ny):
        y=grid_size/2+j*grid_size
        for i in range(nx):
            x=grid_size/2.+float(i)*grid_size
            count = np.abs(np.sum(density_pred[j*grid_size:j*grid_size+grid_size, i*grid_size:i*grid_size+grid_size]))
            if count > 0.1:
                ax.text(x,y,'{:.1f}'.format(count).lstrip('0'),color='red',ha='center',va='center', fontsize=25, alpha=0.8)
    
    canvas.draw()
    data = np.array(canvas.renderer.buffer_rgba())
    plt.close()
    
    return cv2.cvtColor(data, cv2.COLOR_RGBA2RGB)


def visualize_gt_head_feet(img, gt_view, display=False):
    
    img = img.copy()
    
    for gt in gt_view:
        color = (gt.id * 67 % 255, (gt.id + 1) * 36 % 255 , 167)
        
        cv2.drawMarker(img, tuple(gt.feet), color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.drawMarker(img, tuple(gt.head), color, markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)

    if display:    
        plt.imshow(img)
        plt.show()

    return img
