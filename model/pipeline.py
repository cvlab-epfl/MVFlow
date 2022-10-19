import time

import torch

from misc.log_utils import log
from misc.utils import PinnableDict

class MultiViewPipeline(torch.nn.Module):

    def __init__(self, people_flow):
        super(MultiViewPipeline, self).__init__()

        self.people_flow = people_flow

    def forward(self, input_data):
        time_stat = dict()
        end = time.time()

        #Run people flow
        flow_pred = None
        if self.people_flow is not None:
            flow_pred = self.people_flow(input_data)
        
        time_stat["flow_time"] = time.time() - end
        end = time.time()

        return PinnableDict({"flow_pred": flow_pred, "time_stats":time_stat})
