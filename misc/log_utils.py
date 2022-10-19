import collections
import datetime
import logging
import os
import time

from pprint import pformat
from configs import config

config.setup_logging(config.get_logging_dict())
log = logging.getLogger('pose_estimation')  # this is the global logger

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        #TODO add running std to help see convergence

class TimeMeter(AverageMeter):
    def __init__(self):
        super().__init__()
        self.time = time.time()

    def update(self):
        old_time = self.time
        self.time = time.time()
        super().update(self.time - old_time)

class DictMeter(object):
    """ 
        Wrapper function used to track multiple metric stored inside dictionnary
    """

    def __init__(self):
        self.dict_meter = collections.defaultdict(AverageMeter)

    def update(self, dict_metric, n=1):
        for k, v in dict_metric.items():
            self.dict_meter[k].update(v, n=n)

    def avg(self):
        avg_dict = dict()
        for k, v in self.dict_meter.items():
            avg_dict[k] = v.avg

        return avg_dict
    
    def sum(self):
        avg_dict = dict()
        for k, v in self.dict_meter.items():
            avg_dict[k] = v.sum

        return avg_dict

    def __getitem__(self, key):
        if key not in self.dict_meter:
            log.warning(f"Trying to query a key which doesn't exist in the Meter dictionary, returning default 0 value: {key}")

        return self.dict_meter[key]

    def __contains__(self, item):
        return item in self.dict_meter

    def keys(self):
        return self.dict_meter.keys()

def avg_stat_dict(list_of_dict):
    results = collections.defaultdict(int)

    for d in list_of_dict:
        for k, v in d.items():
            results[k] += v / len(list_of_dict)

    return results

def batch_logging(epoch, batch_index, nb_batch, stats_meter, loss_to_print=[], metric_to_print=[], validation=False):
    from misc import metric
    
    tab = '\t'
    #Epoch and step logging
    str_step_lgd = "[Epoch][Bacth/Total]"
    str_step = f"[{epoch}][{batch_index}/{nb_batch}]"

    #Time logging
    batch_time = stats_meter["batch_time"].avg
    data_time = stats_meter["data_time"].avg
    flow_time = stats_meter["flow_time"].avg
    criterion_time = stats_meter["criterion_time"].avg
    optim_time = stats_meter["optim_time"].avg
    
    time_percent_data = int((data_time / batch_time)*100.0)
    time_percent_flow = int((flow_time / batch_time)*100.0)
    time_percent_criterion = int((criterion_time / batch_time)*100.0)
    time_percent_optim = int((optim_time / batch_time)*100.0)

    time_til_end = str(datetime.timedelta(seconds = batch_time * (nb_batch - batch_index))).split(".")[0]

    str_time_lgd = "Time Batch(Data,Flow,Crit,Optim)"
    str_time = f"Time  {batch_time:.2f}({time_percent_data:02d}%,{time_percent_flow:02d}%,{time_percent_criterion:02d}%,{time_percent_optim:02d}%)\tEnd {time_til_end}"

    #Loss logging
    losses = stats_meter["loss"].avg
    loss_component = [stats_meter[loss_c].avg for loss_c in loss_to_print]

    max_char_len = [max(len(lossn), len(f'{lossc:.4f}')) for lossn, lossc in zip(loss_to_print, loss_component)]

    str_loss_lgd = f"Loss \t{'total':<{len(f'{losses:.4f}')}} ({'  '.join([f'{lossn:<{padding}}' for lossn, padding in zip(loss_to_print, max_char_len)])})"
    str_loss = f"Loss  \t{losses:<{len(f'{losses:.4f}')}.4f} ({'  '.join([f'{loss:<{padding}.4f}' for loss, padding in zip(loss_component, max_char_len)])})"
    
    if batch_index == 0:
        log.info("\t".join([str_step_lgd, str_time_lgd, str_loss_lgd]))

    log.info("\t".join([str_step, str_time, str_loss]))

    #During validation also output log metrics
    if validation:
        
        metrics = [stats_meter[metric].avg for metric in metric_to_print]
        
        max_char_len = [max(len(metrn), len(f'{metrc:.3f}')) for metrn, metrc in zip(metric_to_print, metrics)]

        str_metric_lgd = f"Metric {'  '.join([f'{metr:<{padding}}' for metr, padding in zip(metric_to_print, max_char_len)])}"
        str_metric = f"Metric {'  '.join([f'{metric:<{padding}.3f}' for metric, padding in zip(metrics, max_char_len)])}"

        if batch_index == 0:
            log.info("\t" + str_metric_lgd)

        log.info("\t" + str_metric)


def log_epoch(logger, log_dict, epoch):
    flatten_log_dict = flatten_dict(log_dict)
    for k, v in flatten_log_dict.items():
        logger.add_scalar(k, v, epoch)


def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def dict_to_string(dict):
    return pformat(dict)


def set_log_level(log_level_name):
    numeric_level = getattr(logging, log_level_name.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)

    for handler in log.handlers:
        handler.setLevel(numeric_level)
