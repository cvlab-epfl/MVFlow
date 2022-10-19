import numpy as np
import torch

class MSEwithROILoss(torch.nn.Module):
    def __init__(self, reweighting_factor):
        super(MSEwithROILoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction="none")
        self.reweighting_factor = reweighting_factor

    def forward(self, pred, target, roi_mask, reweighting=None):
        loss = self.mse(pred, target) * roi_mask

        if reweighting is not None:
            loss = loss + self.reweighting_factor*loss*reweighting

        return loss.sum()

class ConsistencywithROILoss(torch.nn.Module):
    def __init__(self, reweighting_factor):
        super(ConsistencywithROILoss, self).__init__()

        self.reweighting_factor = reweighting_factor
        self.loss = torch.nn.MSELoss(reduction="none")

    def forward(self, pred1, pred2, roi_mask, reweighting=None):
        
        
        loss = self.loss(torch.clamp(pred1, 1e-10, 1-1e-10), torch.clamp(pred2, 1e-10, 1-1e-10))  * roi_mask
       
        if reweighting is not None:
            loss = loss + self.reweighting_factor*loss*reweighting

        return loss.sum()


class FlowLossProb(torch.nn.Module):
    def __init__(self, data_spec, loss_spec, stats_name="flow"):
        super(FlowLossProb, self).__init__()
 
        self.criterion = MSEwithROILoss(loss_spec["reweigthing_factor"])
        self.consitency_criterion = ConsistencywithROILoss(loss_spec["reweigthing_factor"])

        self.stats_name = stats_name

    def compute_flow_consistency_loss(self, flow, flow_inverse, roi_mask, boundary_mask, reweighting):

        if reweighting is not None:
            loss_consistency = self.consitency_criterion(flow[:,0,1:,1:], flow_inverse[:,8,:-1,:-1], roi_mask[:,0,1:,1:], reweighting=reweighting[:,0,1:,1:]) \
                + self.consitency_criterion(flow[:,1,1:,:], flow_inverse[:,7,:-1,:], roi_mask[:,0,1:,:], reweighting=reweighting[:,0,1:,:]) \
                + self.consitency_criterion(flow[:,2,1:,:-1], flow_inverse[:,6,:-1,1:], roi_mask[:,0,1:,:-1], reweighting=reweighting[:,0,1:,:-1]) \
                + self.consitency_criterion(flow[:,3,:,1:], flow_inverse[:,5,:,:-1], roi_mask[:,0,:,1:], reweighting=reweighting[:,0,:,1:]) \
                + self.consitency_criterion(flow[:,4,:,:], flow_inverse[:,4,:,:], roi_mask[:,0,:,:], reweighting=reweighting[:,0,:,:]) \
                + self.consitency_criterion(flow[:,5,:,:-1], flow_inverse[:,3,:,1:], roi_mask[:,0,:,:-1], reweighting=reweighting[:,0,:,:-1]) \
                + self.consitency_criterion(flow[:,6,:-1,1:], flow_inverse[:,2,1:,:-1], roi_mask[:,0,:-1,1:], reweighting=reweighting[:,0,:-1,1:]) \
                + self.consitency_criterion(flow[:,7,:-1,:], flow_inverse[:,1,1:,:], roi_mask[:,0,:-1,:], reweighting=reweighting[:,0,:-1,:]) \
                + self.consitency_criterion(flow[:,8,:-1,:-1], flow_inverse[:,0,1:,1:], roi_mask[:,0,:-1,:-1], reweighting=reweighting[:,0,:-1,:-1])
        else:
            loss_consistency = self.consitency_criterion(flow[:,0,1:,1:], flow_inverse[:,8,:-1,:-1], roi_mask[:,0,1:,1:]) \
                + self.consitency_criterion(flow[:,1,1:,:], flow_inverse[:,7,:-1,:], roi_mask[:,0,1:,:]) \
                + self.consitency_criterion(flow[:,2,1:,:-1], flow_inverse[:,6,:-1,1:], roi_mask[:,0,1:,:-1]) \
                + self.consitency_criterion(flow[:,3,:,1:], flow_inverse[:,5,:,:-1], roi_mask[:,0,:,1:]) \
                + self.consitency_criterion(flow[:,4,:,:], flow_inverse[:,4,:,:], roi_mask[:,0,:,:]) \
                + self.consitency_criterion(flow[:,5,:,:-1], flow_inverse[:,3,:,1:], roi_mask[:,0,:,:-1]) \
                + self.consitency_criterion(flow[:,6,:-1,1:], flow_inverse[:,2,1:,:-1], roi_mask[:,0,:-1,1:]) \
                + self.consitency_criterion(flow[:,7,:-1,:], flow_inverse[:,1,1:,:], roi_mask[:,0,:-1,:]) \
                + self.consitency_criterion(flow[:,8,:-1,:-1], flow_inverse[:,0,1:,1:], roi_mask[:,0,:-1,:-1])

        return loss_consistency

    def compute_boundary_consistency(self, flow, flow_inverse, mask_boundary, roi_mask):
        exiting_consistency = self.consitency_criterion(flow[0,10], flow_inverse[0,9], roi_mask*mask_boundary)

        return exiting_consistency


    def forward(self, input_data, output_flow):
        roi_mask = input_data["ROI_mask"]
        boundary_mask = input_data["ROI_boundary_mask"]

        hm_1 = input_data["hm_1"]
        hm_0 = input_data["hm_0"]
        hm_2 = input_data["hm_2"]

        post_reweighting = torch.abs(hm_1 - hm_2)
        pre_reweighting = torch.abs(hm_0 - hm_1)

        stats = {}

        loss_rec_1pf = self.criterion(output_flow["rec_1pf"], hm_1, roi_mask, reweighting=pre_reweighting) / 4
        loss_rec_1pb = self.criterion(output_flow["rec_1pb"], hm_1, roi_mask, reweighting=pre_reweighting) / 4
        loss_rec_1nf = self.criterion(output_flow["rec_1nf"], hm_1, roi_mask, reweighting=post_reweighting) / 4
        loss_rec_1nb = self.criterion(output_flow["rec_1nb"], hm_1, roi_mask, reweighting=post_reweighting) / 4

        loss_rec_1o = loss_rec_1pf + loss_rec_1pb + loss_rec_1nf + loss_rec_1nb

        stats = {**stats,
            self.stats_name + "_loss_rec_1pf" : loss_rec_1pf.item(),
            self.stats_name + "_loss_rec_1pb" : loss_rec_1pb.item(),
            self.stats_name + "_loss_rec_1nf" : loss_rec_1nf.item(),
            self.stats_name + "_loss_rec_1nb" : loss_rec_1nb.item(),
            self.stats_name + "_loss_rec_1o" : loss_rec_1o.item()
            }


        loss_rec_0f = self.criterion(output_flow["rec_0f"], hm_0, roi_mask, reweighting=pre_reweighting) / 2 
        loss_rec_0b = self.criterion(output_flow["rec_0b"], hm_0, roi_mask, reweighting=pre_reweighting) / 2
        # loss_prev_inverse = 0
        loss_rec_2f = self.criterion(output_flow["rec_2f"], hm_2, roi_mask, reweighting=post_reweighting) / 2
        loss_rec_2b = self.criterion(output_flow["rec_2b"], hm_2, roi_mask, reweighting=post_reweighting) / 2
        # loss_post_inverse = 0

        loss_rec_0o = loss_rec_0f + loss_rec_0b
        loss_rec_2o = loss_rec_2f + loss_rec_2b


        stats = {**stats,
            self.stats_name + "_loss_rec_0f" : loss_rec_0f.item(),
            self.stats_name + "_loss_rec_0b" : loss_rec_0b.item(),
            self.stats_name + "_loss_rec_2f" : loss_rec_2f.item(),
            self.stats_name + "_loss_rec_2b" : loss_rec_2b.item(),
            self.stats_name + "_loss_rec_0o" : loss_rec_0o.item(),
            self.stats_name + "_loss_rec_2o" : loss_rec_2o.item()
            }

        #Flow temporal consistency loss
        loss_consistency_flow_0_1 = self.compute_flow_consistency_loss(output_flow["flow_0_1f"], output_flow["flow_1_0b"], roi_mask, boundary_mask, reweighting=None) / 9
        loss_consistency_flow_1_2 = self.compute_flow_consistency_loss(output_flow["flow_1_2f"], output_flow["flow_2_1b"], roi_mask, boundary_mask, reweighting=None) / 9

        loss_consistency = loss_consistency_flow_0_1 + loss_consistency_flow_1_2

        stats = {**stats,
            self.stats_name + "_loss_consistency_flow_0_1" : loss_consistency_flow_0_1.item(),
            self.stats_name + "_loss_consistency_flow_1_2" : loss_consistency_flow_1_2.item(),
            self.stats_name + "_loss_consistency" : loss_consistency.item()
            }

        total_loss = loss_rec_1o + loss_rec_0o + loss_rec_2o + loss_consistency 


        stats = {**stats,
            "loss_"+self.stats_name : total_loss.item(),
            }

        return {"loss":total_loss, "stats":stats}


class MultiLoss(torch.nn.Module):
    def __init__(self, flow_loss):
        super(MultiLoss, self).__init__()

        self.flow_loss = flow_loss

    def forward(self, input_data, output):

        loss = 0
        stats = {}

        if self.flow_loss is not None:
            flow = self.flow_loss(input_data, output["flow_pred"])
            loss = loss + flow["loss"]
            stats = {**stats, **flow["stats"]}

        stats = {**stats, "loss":loss}

        return {"loss":loss, "stats":stats}