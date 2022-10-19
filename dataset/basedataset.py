import random

import numpy as np
import torch
import torchvision

from collections import defaultdict
from torchvision import transforms


from augmentation.homographyaugmentation import HomographyDataAugmentation
from dataset.utils import is_in_frame, extract_points_from_gt, generate_flow, aggregate_multi_view_gt_points, generate_scene_roi_from_view_rois
from dataset.heatmapbuilder import constant_center_heatmap
from misc import geometry
from misc.utils import PinnableDict, stack_tensors, listdict_to_dictlist
from misc.log_utils import log



class FlowSceneSet(torch.utils.data.Dataset):
    
    def __init__(self, scene_set, data_conf, use_augmentation, compute_flow_stat=False):
        self.scene_set = scene_set
        
        self.nb_view = len(data_conf["view_ids"])
        self.nb_frames = data_conf["nb_frames"]
        self.frame_interval = data_conf["frame_interval"]
        self.generate_flow_hm = False

        log.debug(f"Flow scene set containing {self.nb_view} view and {len(self.scene_set)} frames, will process {self.nb_frames} frame at a time")

        #original image and gt dimension needed for further rescaling
        self.frame_original_size = self.scene_set.frame_original_size
        self.frame_input_size = data_conf["frame_input_size"]

        #homography information are needed to generate hm
        self.homography_input_size = data_conf["homography_input_size"]
        self.homography_output_size = data_conf["homography_output_size"]

        self.hm_builder = data_conf["hm_builder"]
        self.hm_radius = data_conf["hm_radius"]
        self.hm_size = data_conf["hm_size"]
        self.hm_image_size = data_conf["hm_image_size"]
        
        #Reduce length by two to be able to return triplet of frames
        self.total_number_of_frame = len(self.scene_set) - (self.nb_frames-1)*self.frame_interval
        
        self.img_transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             transforms.Resize(self.frame_input_size)
            ]
        )

        self.use_augmentation = use_augmentation

        #list of augmentation and their probability
        self.view_based_augs = [
            (HomographyDataAugmentation(torchvision.transforms.RandomResizedCrop(self.frame_input_size)), 0.5),
            (None, 0.5)
            ]
        self.scene_base_augs = [
            (HomographyDataAugmentation(torchvision.transforms.RandomAffine(degrees = 45, translate = (0.2, 0.2), scale = (0.8,1.2), shear = 10)), 0.5),
            (None, 0.5)
            ]

        if sum([x[1] for x in self.scene_base_augs]) != 1:
            log.warning(f"Scene based augmentation probability should sum up to one but is {sum([x[1] for x in self.scene_base_augs])}")
        if sum([x[1] for x in self.view_based_augs]) != 1:
            log.warning(f"View based augmentation probability should sum up to one but is {sum([x[1] for x in self.view_based_augs])}")

        if compute_flow_stat:
            self.compute_flow_stat()

        
    def __getitem__(self, index):
        multi_view_data = list()
        
        scene_base_aug = self.select_augmentation(*zip(*self.scene_base_augs))
        ROIs_view = list()

        for view_id in range(self.nb_view):
            frame_dict = {}
            frame_dict["view_id"] = view_id

            view_based_aug = self.select_augmentation(*zip(*self.view_based_augs))

            for frame_id in range(self.nb_frames):
                true_frame_id = index+frame_id*self.frame_interval
                frame, homography = self.scene_set.get(true_frame_id, view_id)

                #normalize and rescale frame
                frame = self.img_transform(frame)

                #TODO: get individual ROI and occluded area for each frame
                homography = torch.from_numpy(homography).float()

                gt_points_image, person_id_image = self.scene_set.get_gt_image(true_frame_id, view_id)

                frame, homography, gt_points_image, person_id_image = self.apply_view_based_augmentation(frame, homography, gt_points_image, person_id_image, view_based_aug)
                
                hm_image  = self.build_heatmap(gt_points_image, self.hm_image_size, self.hm_radius).squeeze(0)

                frame_dict[f"frame_{frame_id}"] = frame
                frame_dict[f"frame_{frame_id}_true_id"] = true_frame_id
                frame_dict[f"hm_image_{frame_id}"] = hm_image
                frame_dict[f"gt_points_image_{frame_id}"] = gt_points_image
                frame_dict[f"person_id_image_{frame_id}"] = person_id_image


            homography = self.apply_scene_based_aug(homography, scene_base_aug)

            frame_dict["homography"] = homography
            multi_view_data.append(frame_dict)

            if view_based_aug is not None:
                ROIs_view.append(view_based_aug.augment_gt_point_view_based(self.scene_set.get_ROI(view_id), None, filter_out_of_frame=False)[0])
            else:
                ROIs_view.append(self.scene_set.get_ROI(view_id))

        # log.debug(multi_view_data)
        multi_view_data = listdict_to_dictlist(multi_view_data)
        multi_view_data = stack_tensors(multi_view_data)

        #adding groundtuth shared between the view
        for frame_id in range(self.nb_frames):
            true_frame_id = index+frame_id*self.frame_interval

            gt_points, person_id = aggregate_multi_view_gt_points(multi_view_data[f"gt_points_image_{frame_id}"], multi_view_data[f"person_id_image_{frame_id}"], multi_view_data["homography"], self.hm_image_size, self.homography_input_size, self.homography_output_size, self.hm_size)
            gt_points = np.rint(gt_points)

            hm  = self.build_heatmap(gt_points, self.hm_size, self.hm_radius)

            multi_view_data[f"hm_{frame_id}"] = hm
            multi_view_data[f"gt_points_{frame_id}"] = gt_points
            multi_view_data[f"person_id_{frame_id}"] = person_id

        # ROI_mask, ROI_boundary = self.scene_set.get_scene_ROI()
        ROI_mask, ROI_boundary = generate_scene_roi_from_view_rois(ROIs_view, multi_view_data["homography"], self.frame_original_size, self.homography_input_size, self.homography_output_size, self.hm_size)
        ROI_mask = torch.from_numpy(ROI_mask).float().unsqueeze(0)
        boundary_mask = torch.from_numpy(ROI_boundary).float().unsqueeze(0)  

        #adding additional scene data (roi, and boundary)
        multi_view_data["ROI_mask"] = ROI_mask
        multi_view_data["ROI_boundary_mask"] = boundary_mask
        multi_view_data["scene_id"] = self.scene_set.scene_id

        return multi_view_data

    def select_augmentation(self, aug_list, aug_prob):
        if self.use_augmentation:
            aug = random.choices(aug_list, weights=aug_prob)[0]
        else:
            aug = None

        if aug is not None:
            aug.reset()

        return aug

    def apply_view_based_augmentation(self, frame, homography, gt_points_image, person_id_image, view_based_aug):
        if view_based_aug is None:
            return frame, homography, gt_points_image, person_id_image

        frame = view_based_aug(frame)
        homography = view_based_aug.augment_homography_view_based(homography, self.homography_input_size)
        gt_points_image, person_id_image = view_based_aug.augment_gt_point_view_based(gt_points_image, person_id_image)

        return frame, homography, gt_points_image, person_id_image

    def apply_scene_based_aug(self, homography, scene_base_aug):
        if scene_base_aug is None:
            return homography
        
        homography = scene_base_aug.augment_homography_scene_based(homography, self.homography_output_size)

        return homography
    
    def build_heatmap(self, gt_points, hm_size, hm_radius):
        
        if len(gt_points) != 0:
            gt_points = np.rint(gt_points).astype(int)
        hm = self.hm_builder(hm_size, gt_points, hm_radius)
        
        return hm.unsqueeze(0)

    def build_static_mask(self, points_flow, prob_masking):
        
        filtered_points = [point[0] for point in points_flow if point[1] == 4 and random.uniform(0, 1) > prob_masking]

        return constant_center_heatmap(self.hm_size, filtered_points, self.hm_radius+1, value=0, background="one")
            

    def augment_input_data(self, multi_view_data, aug_transform):

        if aug_transform is not None:
            for input_data in multi_view_data:
                input_data["frame"] = aug_transform(input_data["frame"]).contiguous()
                input_data["pre_frame"] = aug_transform(input_data["pre_frame"]).contiguous()
                input_data["post_frame"] = aug_transform(input_data["post_frame"]).contiguous()

                # input_data["hm"] = aug_transform.transforms[0](input_data["hm"]).contiguous()


        return multi_view_data
    
    def disturb_points(self, gt_points):
        
        disturbed_points = list()

        #Leave pre hm empty sometime
        if np.random.random() > 0.1:
            for point in gt_points:

                #Drop gt point to generate False negative
                if np.random.random() > self.pre_hm_fp_rate:
                    disturbed_points.append(point)

                #Add False positive near existing point
                if np.random.random() < self.pre_hm_fn_rate:
                    point_fn = point.copy()
                    point_fn[0] = point_fn[0] + np.random.randn() * 0.05 * self.hm_size[0]
                    point_fn[1] = point_fn[1] + np.random.randn() * 0.05 * self.hm_size[1]
                    
                    disturbed_points.append(point_fn)

        disturbed_points = np.array(disturbed_points)

        return disturbed_points

    def build_tracking_gt(self, gt, pre_gt, pre_gt_points, pre_gt_point_mask, gt_points, gt_point_mask):

        #Find all corespondence between pre point and current point
        person_id = np.array([ann.id for ann in gt])
        pre_person_id = np.array([ann.id for ann in pre_gt])

        intersect_id, ind, pre_ind = np.intersect1d(person_id, pre_person_id, return_indices=True)

        #Initialize map and mask
        tracking_map = np.zeros((2, self.hm_size[0], self.hm_size[1]))
        tracking_mask = np.zeros((1, self.hm_size[0], self.hm_size[1]))

        for i in range(intersect_id.shape[0]):
            frame_id = ind[i]
            pre_id = pre_ind[i]

            #filter out out of frame gt points
            if gt_point_mask[frame_id] and pre_gt_point_mask[pre_id]:
                frame_id = np.sum(gt_point_mask[:frame_id].astype(int))
                pre_id = np.sum(pre_gt_point_mask[:pre_id].astype(int))
            else:
                #if person hidden in frame or previous frame we skip it
                continue
            
            point = gt_points[frame_id].astype(int)
            tracking_map[:, point[1], point[0]] = gt_points[frame_id] - pre_gt_points[pre_id]
            tracking_mask[:, point[1], point[0]] = 1


        tracking_map = torch.from_numpy(tracking_map).to(torch.float32)
        tracking_mask = torch.from_numpy(tracking_mask).to(torch.float32)

        return tracking_map, tracking_mask

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
    
    def __len__(self):
        return self.total_number_of_frame

    @staticmethod
    def collate_fn(batch):
        
        #Merge dictionnary
        batch = listdict_to_dictlist(batch)
        batch = stack_tensors(batch)

        collate_dict = PinnableDict(batch)

        log.spam(f"collate_dict {type(collate_dict)}")

        return collate_dict


    def compute_flow_stat(self):
        ROI_mask, ROI_boundary = self.scene_set.get_scene_ROI()

        flow_count_per_channel = defaultdict(int) 
        total_move_length = list()
        nb_overlaping_gt = 0
        for index in range(self.total_number_of_frame):
            index_next = index + self.frame_interval

            gt_points, person_id = extract_points_from_gt(self.scene_set.get_gt(index), self.hm_size)
            next_gt_points, next_person_id = extract_points_from_gt(self.scene_set.get_gt(index_next), self.hm_size)

            gt_points = np.rint(gt_points).astype(int)
            next_gt_points = np.rint(next_gt_points).astype(int)

            nb_overlaping_gt += gt_points.shape[0]-np.unique(gt_points, axis=0).shape[0]
            _, flow_gt_list, move_length = generate_flow(gt_points, person_id, next_gt_points, next_person_id, ROI_mask, hm_radius=-1, generate_hm=False)
            
            total_move_length.extend(move_length)
                       
            for (pos, flow_channel) in flow_gt_list:
                flow_count_per_channel[flow_channel] += 1

        
        for k, v in flow_count_per_channel.items():
            if v != 0:
                log.info(f"flow stats channel {k} : {v}")

        log.info(f"Motion of len 0: {len([x for x in total_move_length if x < 1])}")
        log.info(f"Motion of len 1: {len([x for x in total_move_length if x < 2 and x >= 1])}")
        log.info(f"Motion of len 2: {len([x for x in total_move_length if x < 3 and x >= 2])}")
        log.info(f"Motion of len 3: {len([x for x in total_move_length if x < 4 and x >= 3])}")
        log.info(f"Motion of len 4: {len([x for x in total_move_length if x < 5 and x >= 4])}")
        log.info(f"Motion of len >5: {len([x for x in total_move_length if x >= 5])}")

        log.info(f"Number of person in the gt overlapping {nb_overlaping_gt}")