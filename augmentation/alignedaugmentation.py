import math 

import torch

from torchvision.transforms.functional import _get_perspective_coeffs, vflip, hflip

from augmentation.reapeatabletransform import RepeatableTransform


class AlignedResizedCropTransform(RepeatableTransform):
    def __init__(self, resized_crop):
        super().__init__(resized_crop)
    
    def _get_aug_transform_matrix(self):
        #crop can be seen as a combination of scaling and translation
        crop_t, crop_l, crop_h, crop_w = self.last_params
        
        #Scaling matrix
        S = torch.eye(3)
        S[0,0] = self.img_w_out / crop_w
        S[1,1] = self.img_h_out / crop_h
        
        #Translation matrix
        T = torch.eye(3)
        T[0,2] = -crop_l
        T[1,2] = -crop_t

        
        aug_mat = S @ T 
        
        return aug_mat


class AlignedAffineTransform(RepeatableTransform):
    def __init__(self, resized_crop):
        super().__init__(resized_crop)
    
    def _get_aug_transform_matrix(self):
        #crop can be seen as a combination of scaling and translation
        angle, translations, scale, shear = self.last_params
        
        #Pytorch code convert param to homography matrix
        rot = math.radians(angle)
        sx = math.radians(shear[0])
        sy = math.radians(shear[1])

        cx, cy = self.img_w_in*0.5, self.img_h_in*0.5
        tx, ty = translations

        # RSS without scaling
        a = math.cos(rot - sy) / math.cos(sy)
        b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
        c = math.sin(rot - sy) / math.cos(sy)
        d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

        matrix = [a, b, 0.0, c, d, 0.0]
        matrix = [x * scale for x in matrix]
        # Apply inverse of center translation: RSS * C^-1
        matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
        matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
        # Apply translation and center : T * C * RSS * C^-1
        matrix[2] += cx + tx
        matrix[5] += cy + ty
        
        aug_mat = torch.eye(3)
        
        aug_mat[0:2] = torch.tensor(matrix).reshape(2,3)
        
        return aug_mat

 
class AlignedPerspectiveTransform(RepeatableTransform):
    def __init__(self, resized_crop):
        super().__init__(resized_crop)
    
    def _get_aug_transform_matrix(self):
        #crop can be seen as a combination of scaling and translation
        startpoints, endpoints = self.last_params
        
        coeffs = _get_perspective_coeffs(endpoints, startpoints)
        
        aug_mat = torch.eye(3).reshape(-1)
        aug_mat[0:8] = torch.tensor(coeffs)
        aug_mat = aug_mat.reshape(3,3)
        
        return aug_mat


class AlignedVerticalFlipTransform(RepeatableTransform):
    def __init__(self, resized_crop):
        super().__init__(resized_crop)
    
    def forward(self, img):
        self.img_h_in, self.img_w_in = img.shape[-2:]
        
        if self.last_params is None:
            self.last_params = torch.rand(1) < self.transform.p
        
        if self.last_params:
            return vflip(img)
        
        return img
    
    def _get_aug_transform_matrix(self):
        aug_mat = torch.eye(3)
        
        if self.last_params:
            aug_mat[1,1] = -1
            aug_mat[1,2] = self.img_h_in
            
        return aug_mat


class AlignedHorizontalFlipTransform(RepeatableTransform):
    def __init__(self, resized_crop):
        super().__init__(resized_crop)
    
    def forward(self, img):
        self.img_h_in, self.img_w_in = img.shape[-2:]
        
        if self.last_params is None:
            self.last_params = torch.rand(1) < self.transform.p
        
        if self.last_params:
            return hflip(img)
        
        return img
    
    def _get_aug_transform_matrix(self):
        aug_mat = torch.eye(3)
        
        if self.last_params:
            aug_mat[0,0] = -1
            aug_mat[0,2] = self.img_w_in
            
        return aug_mat