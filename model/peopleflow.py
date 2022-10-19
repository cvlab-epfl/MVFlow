import cv2

import numpy as np
import torch 
import torch.nn.functional as F 

from misc.log_utils import log

from model.multimodel import MultiNet

class PeopleFlowProb(torch.nn.Module):
    def __init__(self, data_spec):
        super().__init__()


        self.nb_hm = 10

        # self.ground_hm_size = data_spec["hm_size"]#
        self.homography_input_size = data_spec["homography_input_size"]
        self.homography_output_size = data_spec["homography_output_size"]

        self.hm_size = data_spec["hm_size"]

        self.flow_model = MultiNet(
            self.hm_size, 
            self.homography_input_size, 
            self.homography_output_size, 
            nb_ch_out=self.nb_hm,
            nb_view=len(data_spec["view_ids"])
            )


    def reconstruct_from_prev(self, prev_flow, mask_boundry):

        #aligning all the flow
        prev_flow = torch.stack([
            F.pad(prev_flow[:,0,1:,1:], (0,1,0,1)), 
            F.pad(prev_flow[:,1,1:,:], (0,0,0,1)), 
            F.pad(prev_flow[:,2,1:,:-1], (1,0,0,1)), 
            F.pad(prev_flow[:,3,:,1:], (0,1,0,0)), 
            prev_flow[:,4,:,:], 
            F.pad(prev_flow[:,5,:,:-1], (1,0,0,0)), 
            F.pad(prev_flow[:,6,:-1,1:], (0,1,1,0)), 
            F.pad(prev_flow[:,7,:-1,:], (0,0,1,0)), 
            F.pad(prev_flow[:,8,:-1,:-1], (1,0,1,0)), 
            prev_flow[:,9,:,:] * mask_boundry.squeeze(1), #entering scene flow
        ], dim = 1)

        reconstruction_from_prev = torch.sum(prev_flow[:,:10,:,:], dim=1, keepdim=True)

        return reconstruction_from_prev

    def reconstruct_from_post(self, post_flow):

        reconstruction_from_post = torch.sum(post_flow[:,:9,:,:], dim=1, keepdim=True)

        return reconstruction_from_post

    def reconstruct_from_flow(self, flow_0_1f, flow_1_2f, flow_1_0b, flow_2_1b, mask_boundry, type=""):
        
        #reconstruction of current frame prediction
        rec_1pf = self.reconstruct_from_prev(flow_0_1f, mask_boundry)
        rec_1pb = self.reconstruct_from_post(flow_1_0b)

        rec_1nf = self.reconstruct_from_post(flow_1_2f)
        rec_1nb = self.reconstruct_from_prev(flow_2_1b, mask_boundry)

        #reconstruction of prev frame prediction        
        rec_0f = self.reconstruct_from_post(flow_0_1f)
        rec_0b = self.reconstruct_from_prev(flow_1_0b, mask_boundry)

        #reconstruction of post frame prediction        
        rec_2f = self.reconstruct_from_prev(flow_1_2f, mask_boundry)
        rec_2b = self.reconstruct_from_post(flow_2_1b)

        rec_1o = (rec_1pf + rec_1pb + rec_1nf + rec_1nb) / 4

        reconstructions = {
            f"rec_1pf{type}" : rec_1pf,
            f"rec_1pb{type}" : rec_1pb,
            f"rec_1nf{type}" : rec_1nf,
            f"rec_1nb{type}" : rec_1nb,
            f"rec_0f{type}" : rec_0f,
            f"rec_0b{type}" : rec_0b,
            f"rec_2f{type}" : rec_2f,
            f"rec_2b{type}" : rec_2b,
            f"rec_1o{type}" : rec_1o.detach()
        }

        return reconstructions


    def forward(self, input_data):

        flow_0_1f, flow_1_0b, = self.flow_model(input_data["frame_0"], input_data["frame_1"], input_data["homography"], input_data["ROI_mask"])
        flow_1_2f, flow_2_1b, = self.flow_model(input_data["frame_1"], input_data["frame_2"], input_data["homography"], input_data["ROI_mask"])


        roi_mask = input_data["ROI_mask"]
        mask_boundry = input_data["ROI_boundary_mask"]

        reconstructions = self.reconstruct_from_flow(flow_0_1f, flow_1_2f, flow_1_0b, flow_2_1b, mask_boundry)

        output = {
            "flow_0_1f":flow_0_1f,
            "flow_1_2f":flow_1_2f,
            "flow_1_0b":flow_1_0b,
            "flow_2_1b":flow_2_1b,
            **reconstructions
        }


        return output