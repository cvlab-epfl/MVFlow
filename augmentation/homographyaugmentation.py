
import numpy as np
import torch
import torchvision

import augmentation.alignedaugmentation as alaug 
from dataset.utils import is_in_frame
from misc import geometry

class HomographyDataAugmentation(torch.nn.Module):
    """
    Data augmentation for image, gt, homography structure, which is reapeatable can be applied to multiple set of images, and consistant.
    In case of view based augmentation the homography is updated such that augmentation apply to the image doesn't change the final projection of the image.
    In case of scene based augmentation the homography is updated on it's own.
    """
    def __init__(self, transform):
        super().__init__()
        
        if isinstance(transform, torchvision.transforms.RandomAffine):
            self.aligned_transform = alaug.AlignedAffineTransform(transform)
        elif isinstance(transform, torchvision.transforms.RandomResizedCrop):
            self.aligned_transform = alaug.AlignedResizedCropTransform(transform)
        elif isinstance(transform, torchvision.transforms.RandomHorizontalFlip):
            self.aligned_transform = alaug.AlignedHorizontalFlipTransform(transform)  
        elif isinstance(transform, torchvision.transforms.RandomVerticalFlip):
            self.aligned_transform = alaug.AlignedVerticalFlipTransform(transform) 
        elif isinstance(transform, torchvision.transforms.RandomPerspective):
            self.aligned_transform = alaug.AlignedPerspectiveTransform(transform)
        elif transform is None:
            self.aligned_transform=None
        else:
            raise Exception("sorry, we do not support:" + type(transform).__name__)

        self.reset()

    def reset(self):
        self.aligned_transform.reset()

    def forward(self, image):
        if self.aligned_transform is None:
            return image
        
        return self.aligned_transform(image)
    
    def augment_homography_view_based(self, homography, homography_input_size):
        if self.aligned_transform is None:
            return homography
        
        S1 = torch.eye(3)
        S1[0,0] = homography_input_size[1] / self.aligned_transform.img_w_out
        S1[1,1] = homography_input_size[0] / self.aligned_transform.img_h_out
        
        S2 = torch.eye(3)
        S2[0,0] = self.aligned_transform.img_w_in / homography_input_size[1] 
        S2[1,1] = self.aligned_transform.img_h_in / homography_input_size[0] 
        
        homography = S1 @ self.aligned_transform.get_aug_transform_matrix() @ S2 @ homography
        
        return homography
        
    def augment_homography_scene_based(self, homography, homography_output_size):
        if self.aligned_transform is None:
            return homography

        self.aligned_transform.initialize_params(homography_output_size)
        
        homography = homography @ self.aligned_transform.get_aug_transform_matrix()
        
        return homography
        
    def augment_gt_point_view_based(self, gt_points, gt_person_ids, filter_out_of_frame=True):
        if self.aligned_transform is None or len(gt_points) == 0:
            return gt_points, gt_person_ids

        points_aug = geometry.project_points(gt_points, self.aligned_transform.get_aug_transform_matrix().numpy())
        
        if filter_out_of_frame:
            #mask point put out of frame
            mask_visible = np.array([is_in_frame(point, (self.aligned_transform.img_w_out, self.aligned_transform.img_h_out)) for point in points_aug])
            points_aug = points_aug[mask_visible]
            
            if gt_person_ids is not None:
                gt_person_ids = gt_person_ids[mask_visible]
        
        return points_aug, gt_person_ids