from misc.log_utils import log, dict_to_string
from loss import loss


def get_loss(data_spec, loss_spec):
    log.info(f"Building Loss")
    log.debug(f"Loss spec: {dict_to_string(loss_spec)}")

    flow = loss.FlowLossProb(data_spec, loss_spec, "flow")
    
    global_criterion = loss.MultiLoss(flow)

    return global_criterion