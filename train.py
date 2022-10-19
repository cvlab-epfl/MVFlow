import os
import psutil
import time
import warnings

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from configs.arguments import get_config_dict
from dataset import factory as data_factory
from evaluation import Evaluator
from loss import factory as loss_factory
from misc.log_utils import DictMeter, batch_logging, log, dict_to_string, log_epoch
from misc.utils import save_checkpoint, check_for_existing_checkpoint
from model import factory as model_factory

warnings.filterwarnings("ignore", category=UserWarning)

def train(train_loaders, model, criterion, optimizer, epoch, conf):
    stats_meter  = DictMeter()
    model.train()

    total_nb_frames = sum([len(train_loader) for train_loader in train_loaders])

    end = time.time()
    for s, train_loader in enumerate(train_loaders):
        for f, input_data in enumerate(train_loader):
            
            #global index of current frame
            i = sum([len(train_loaders[x]) for x in range(s)]) + f

            input_data = input_data.to(conf["device"])
            data_time = time.time() - end
            
            output_data = model(input_data)
            
            end2 = time.time()
            criterion_output = criterion(input_data, output_data)
            criterion_time = time.time() - end2 

            criterion_output["loss"].backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), conf["training"]["gradient_clip_value"])
            optimizer.step()
            optimizer.zero_grad()

            optim_time = time.time() - end2 - criterion_time

            batch_time = time.time() - end

            epoch_stats_dict = {**criterion_output["stats"], **output_data["time_stats"], "batch_time":batch_time, "data_time":data_time, "criterion_time":criterion_time, "optim_time":optim_time}
            stats_meter.update(epoch_stats_dict)

            if i % conf["main"]["print_frequency"] == 0 or i == (total_nb_frames - 1):
                batch_logging(epoch, i, total_nb_frames, stats_meter, loss_to_print=conf["training"]["loss_to_print"], metric_to_print=conf["training"]["metric_to_print"])

            del input_data
            del output_data
            
            end = time.time()

    return {"stats": stats_meter.avg()}


if __name__ == '__main__':

    #parse arg and config file
    config = get_config_dict()
    log.debug(dict_to_string(config))

    ##################
    ### Initialization
    ##################
    logger = SummaryWriter(os.path.join(config["training"]["ROOT_PATH"] + "/logs/", config["main"]["name"]))

    config["device"] = torch.device('cuda' if torch.cuda.is_available() and config["main"]["device"] == "cuda" else 'cpu') 
    log.info(f"Device: {config['device']}")
    
    start_epoch = 0

    resume_checkpoint = check_for_existing_checkpoint(config["training"]["ROOT_PATH"], config["main"]["name"]) # "model_335")#
    if resume_checkpoint is not None:
        log.info(f"Checkpoint for model {config['main']['name']} found, resuming from epoch {resume_checkpoint['epoch']}")
        # assert resume_checkpoint["conf"] == config
        start_epoch = resume_checkpoint["epoch"] + 1

    end = time.time()
    log.info("Initializing model ...")

    model = model_factory.pipelineFactory(config["data_conf"])

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["state_dict"])

    model.to(config["device"])

    log.info(f"Model initialized in {time.time() - end} s")

    criterion = loss_factory.get_loss(config["data_conf"], config["loss_conf"])

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["training"]["lr"], weight_decay=config["training"]["decay"])
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])

    lr_scheduler = MultiStepLR(optimizer, config["training"]["lrd"][1:], gamma=config["training"]["lrd"][0])

    if resume_checkpoint is not None:
        lr_scheduler.load_state_dict(resume_checkpoint["scheduler"])


    end = time.time()
    log.info("Loading Data ...")

    train_dataloaders, val_dataloaders = data_factory.get_dataloader(config["data_conf"])
    
    log.info(f"Data loaded in {time.time() - end} s")
    ############
    ### Training
    ############
    process = psutil.Process(os.getpid()) 
    
    #Evaluator store tracker processes and encapsulate all the evaluation step
    evaluator = Evaluator(val_dataloaders, model, criterion, 0, config, use_tracker=not(config["training"]["disable_tracker"]))

    end = time.time()
    for epoch in range(start_epoch, config["training"]["max_epoch"]):
        log.info(f"Memory usage {process.memory_info().rss / 1024 / 1024 / 1024} GB")

        log.info(f"{f''' Beginning epoch {epoch} of {config['main']['name']} ''':#^150}")
        train_result = train(train_dataloaders, model, criterion, optimizer, epoch, config)
        train_time = time.time() - end
        # train_result = {"stats":{}}
        log.info(f"{f''' Traning for epoch {epoch} of {config['main']['name']} completed in {train_time:.2f}s ''':#^150}")
        
        log.info(f"{f' Beginning validation for epoch {epoch} ':*^150}")
        valid_results = evaluator.run(epoch)
        log.info(f"{f' Validation for epoch {epoch} completed in {(time.time() - end - train_time):.2f}s ':*^150}")

        log_epoch_dict = {"train":train_result["stats"], "val":valid_results["stats"], "lr":optimizer.param_groups[0]['lr']}

        log_epoch(logger, log_epoch_dict, epoch)
        save_checkpoint(model, optimizer, lr_scheduler, config, log_epoch_dict, epoch, evaluator.is_best)

        lr_scheduler.step()
        log.info(f"Epoch {epoch} completed in {time.time()-end}s")

        end = time.time()

    #Waiting for tracker to complete before exiting
    evaluator.waiting_for_tracker_to_finish()
        
    log.info('Training complete')


# runG "train.py -hmt center -splt 0.9 -dset PETS -vis -bs 1 -flm prob -flagg sum -flc mse -clt mse -hmt constant -hmr 0 -rf 0 -fea r50 -lr 0.001 -fmt multi -nbv 3 -mva minmax -mtl 40 -ugin -cs 224 224"
# python train.py -hmt center -splt 0.9 -dset wild -vis -bs 1 -flm prob -flagg sum -flc mse -clt mse -hmt constant -hmr 0 -rf 0 -fea r50 -lr 0.001 -mcon -fmt multi -mva minmax -nbv 7 -mtl 10 -pf 10 -of -cs 224 224 -n model_test_multibackend2
# runG "train.py -hmt center -splt 0.9 -dset wild -vis -bs 1 -flm prob -flagg sum -flc mse -clt mse -hmt constant -hmr 0 -rf 0 -fea r50 -lr 0.001 -fmt can -nbv 7 -mva minmax -mtl 40 -ugin -cs 224 224"
#train.py -splt 0.9 -dset wild -vis -bs 1 -flm prob -flagg sum -flc mse -clt mse -hmt constant -hmr 0 -rf 5 -fea r34 -lr 0.001 -fmt multi -nbv 7 -mva minmax -mtl 40 -ugin -cs 1080 1920
#train.py -splt 0.6 -dset PETSeval -vis -bs 1 -flm prob -flagg sum -flc mse -clt mse -hmt constant -hmr 0 -rf 5 -fea r34 -lr 0.001 -fmt multi -nbv 7 -mva minmax -mtl 20 -ugin -cs 576 768 
