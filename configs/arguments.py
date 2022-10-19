import sys
import argparse

from configs.pathes import conf_path
from configs.utils import read_yaml_file, convert_yaml_dict_to_arg_list, fill_dict_with_missing_value
from configs.utils import args_to_dict
from misc.log_utils import log, set_log_level


parser = argparse.ArgumentParser(description='Process some integers.')

parser_train = parser.add_argument_group("training")
parser_data = parser.add_argument_group("dataset")
parser_model = parser.add_argument_group("model")
parser_loss = parser.add_argument_group("loss")


####### Configuration #######
parser.add_argument("-n", '--name', default="model", help='Name of the model')
parser.add_argument("-pf", "--print_frequency", dest="print_frequency", help="Number of element processed between print", type=int, default=2500)
parser.add_argument("-dev", "--device", dest="device", help="select device to use either cpu or cuda", default="cuda")
parser.add_argument("-ch", "--nb_checkpoint", dest="nb_checkpoint", default=10, type=int, help="maximum number to checkpoint to save, once reach will start overwrite old checkpoint")
parser.add_argument("-l", "--log_lvl", dest="log_lvl", default="debug", choices=["debug", "spam", "verbose", "info", "warning", "error"], help='Set level of logger to get more or less verbose output')
parser.add_argument("-mtl", "--max_tracklet_lenght", dest="max_tracklet_lenght", help="Number of element processed between print", type=int, default=100)
parser.add_argument("-cfg", "--config_file", dest="config_file", help="Path to a YAML file containing of argument that will be used as new default value, if command line argument are specified they will override the value from the YAML file", type=str, default=None)


####### Training #######
parser_train.add_argument("-lr", "--learning_rate", dest="lr", type=float, default=0.0001, help="Initialization of the learning rate")
parser_train.add_argument("-lrd", "--learning_rate_decrease", dest="lrd", nargs='+', type=float, default=[0.5, 20, 40, 60, 80, 100], help="List of epoch where the learning rate is decreased (multiplied by first arg of lrd)")
parser_train.add_argument("-dec", "--decay", dest="decay", type=float, default=5e-4, help="Adam weight decay")
parser_train.add_argument("-vis", '--eval_visual', dest="eval_visual", action="store_true", default=True, help="Create video visualization from evaluation outputs")
parser_train.add_argument("-mepoch", '--max_epoch', dest="max_epoch", type=int, default=200, help="Max epoch")
parser_train.add_argument("-dtrack", '--disable_tracker', dest="disable_tracker", action="store_true", default=False, help="if flag is used it disable the use of tracker during evaluation")
parser_train.add_argument("-gclip", '--gradient_clip_value', dest="gradient_clip_value", type=float, default=10000000000, help="Value over which the gradient is clipped to stabilize training and avoid gradient explosion")
parser_train.add_argument("-dtv", "--detection_to_evaluate", dest="detection_to_evaluate", nargs='+', type=str, default=["det_0f","det_1b","det_1f","det_2b","rec_1f","rec_0b","rec_2f","rec_1b"], help="List of name of detection to evaluate at inference time")
parser_train.add_argument("-mtv", "--motion_to_evaluate", dest="motion_to_evaluate", nargs='+', type=str, default=[], help="List of name of prediction motion to evaluate at inference time")
parser_train.add_argument("-metp", "--metric_to_print", dest="metric_to_print", nargs='+', type=str, default=["moda_det_1b","recall_det_1b","precision_det_1b","moda_rec_0b","recall_rec_0b","precision_rec_0b"], help="List of metric that will appeared in the validation log file every time a log is made")
parser_train.add_argument("-lott", "--loss_to_print", dest="loss_to_print", nargs='+', type=str, default=["loss_cont_det","loss_cont_rec","loss_cont_time_consistency","loss_cont_reg_offset"], help="List of losses that will appeared in the log file every time a log is made")


####### Data #######
parser_data.add_argument("-dset", "--dataset", dest="dataset", default=["PETS"], nargs='*', choices=["PETS", "PETSeval", "wild", "wildext"], help='Dataset to use for Training')
parser_data.add_argument("-edset", "--eval-dataset", dest="eval-dataset", default=[], nargs='*', choices=["PETS", "PETSeval", "wild", "wildext"], help='Additionnal dataset only used for evaluation purpose')
parser_data.add_argument("-nbf", "--nb_frames", dest="nb_frames", type=int, default=3, help="Number of frame from the same scene to process at the same time")
parser_data.add_argument("-vid", "--view_ids", dest="view_ids", type=int, default=[0], nargs='*', help="Id of the view from the same scene to process at the same time start a 0")
parser_data.add_argument("-hmt", "--hm_type", dest="hm_type", default="constant", choices=["density", "center", "constant"], help='type of methode use to generate groundtruth heatmaps')
parser_data.add_argument("-hms", "--hm_size", dest="hm_size", nargs="+", type=int, default=[128, 128], help="Expect pairs of integer coresponding to the size to resize the image before processing")
parser_data.add_argument("-hmr", "--hm_radius", dest="hm_radius", type=int, default=0,  help="The radius of the gaussian filter used to generate heatmpas")
parser_data.add_argument("-splt", "--split_proportion", dest="split_proportion", type=float, default=0.9, help="Train val split proportion the first split_proportion percent of the frames are used for training, the rest for validation")
parser_data.add_argument("-shft", "--shuffle_train", dest="shuffle_train", action="store_false", default=True, help="By default train set is shuffled, use arg to disable shuffling")
parser_data.add_argument("-bs", "--batch_size", dest="batch_size", type=int, default=1,  help="The size of the batches")
parser_data.add_argument("-dfps", "--desired_fps", dest="desired_fps", type=int, default=-1,  help="The desired framerate use to load the data, must be lower than the original framerate and a factor of it, if set to -1 it will default to the original framerate")
parser_data.add_argument("-cs", "--crop_size", dest="crop_size", nargs="+", type=int, default=[360, 640], help="Expect pairs of integer coresponding to the size to resize the image before processing") #[480, 832]
parser_data.add_argument("-nw", "--num_workers", dest="num_workers", type=int, default=10, help="Number of work to use in the train and val dataloader")
parser_data.add_argument("-aug", "--aug_train", dest="aug_train", action="store_true", default=False, help="Add various data augmentation to input frames during training")
parser_data.add_argument("-fi", "--frame_interval", dest="frame_interval", type=int, default=3, help="Interval between frame when sampling a video to make frame triplets")

####### Loss #######
parser_loss.add_argument("-rf", '--reweigthing_factor', dest="reweigthing_factor", type=float, default=0, help="reweighting factor to weight mistake on prediction for non static person")

def get_config_dict(existing_conf_dict=None):
    """
    Generate config dict from command line argument and config file if existing_conf_dict is not None, value are added to the existing dict if they are not already defined, 
    """

    log.debug(f'Original command {" ".join(sys.argv)}')
    args = parser.parse_args()

    if args.config_file is not None:
        yaml_dict = read_yaml_file(args.config_file)
        arg_list = convert_yaml_dict_to_arg_list(yaml_dict)

        args = parser.parse_args(arg_list + sys.argv[1:])

    args_dict = args_to_dict(parser, args)

    config = {"data_conf": args_dict["dataset"], "loss_conf":args_dict["loss"], "training": {**args_dict["training"], **conf_path}, "main":vars(args)}
    set_log_level(config["main"]["log_lvl"])

    config["data_conf"]["frame_input_size"] = config["data_conf"]["crop_size"]
    config["data_conf"]["homography_input_size"] = [x // 8  for x in config["data_conf"]["frame_input_size"]]
    config["data_conf"]["homography_output_size"] = config["data_conf"]["hm_size"]
    config["data_conf"]["hm_image_size"] = [x // 8  for x in config["data_conf"]["frame_input_size"]]

    if existing_conf_dict is not None:
        #Use existing dict as config, and fill it with value if some of them were missing
        config = fill_dict_with_missing_value(existing_conf_dict, config)

    return config

if __name__ == '__main__':
    conf_dict = get_config_dict()
