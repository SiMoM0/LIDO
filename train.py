import os
import yaml
import torch
import shutil
import random
import datetime
import argparse
import subprocess
import numpy as np
from shutil import copyfile

from modules.trainer import Trainer

def seed_everything(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f'Using seed: {seed}')

def set_torchsparse_kmap_mode():
    import torchsparse.nn.functional as F
    # F.set_kmap_mode("hashmap") # not working properly
    # print(f'Torchsparse kmap mode: {F.get_kmap_mode()}')

    # possible solution as in https://github.com/mit-han-lab/torchsparse/issues/308
    conv_config = F.conv_config.get_default_conv_config(conv_mode=F.get_conv_mode())
    conv_config.kmap_mode = 'hashmap'
    F.conv_config.set_global_conv_config(conv_config)
    print(f'Torchsparse kmap conv mode: {F.conv_config.get_global_conv_config().kmap_mode}')

if __name__ == '__main__':
    seed_everything()
    set_torchsparse_kmap_mode()

    parser = argparse.ArgumentParser("./train.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--config',
        type=str,
        required=False,
        default='config/MinkowskiNet-semantickitti.yaml',
        help='Architecture yaml cfg file. See /config/ for sample.',
    )
    parser.add_argument(
        '--data',
        type=str,
        required=False,
        default='config/labels/semantic-kitti.yaml',
        help='Classification yaml cfg file. See /config/labels for sample. No default!',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=os.getcwd() + '/log/anom' + '/',
        help='Directory to put the log data. Default: ./log/anom'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=False,
        default=None,
        help='File to the checkpoint model to resume training. If not passed, do from scratch!'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=False,
        help='Use pretrained model. If False, do from scratch!'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        default=True,
        help='Use mixed precision training. Default: True'
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("config", FLAGS.config)
    print("data", FLAGS.data)
    print("log", FLAGS.log)
    print("checkpoint", FLAGS.checkpoint)
    print("pretrained", FLAGS.pretrained)
    print("fp16", FLAGS.fp16)
    print("----------\n")
    #print("Commit hash (training version): ", str(
    #    subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
    print("----------\n")

    # open arch config file
    try:
        print("Opening arch config file %s" % FLAGS.config)
        ARCH = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()
    
    # open data config file
    try:
        print("Opening data config file %s" % FLAGS.data)
        DATA = yaml.safe_load(open(FLAGS.data, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()
    
    # create log folder
    try:
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log)
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()

    # does model folder exist?
    if FLAGS.checkpoint is not None:
        if os.path.isfile(FLAGS.checkpoint):
            print("pretrained model found! Using model from %s" % (FLAGS.checkpoint))
        else:
            print("model folder doesnt exist! Start with random weights...")
    else:
        print("No pretrained model found.")

    # copy all files to log folder (to remember what we did, and make inference
    # easier). Also, standardize name to be able to open it later
    try:
        print("Copying files to %s for further reference." % FLAGS.log)
        copyfile(FLAGS.config, FLAGS.log + "/MinkowskiNet-semantickitti.yaml")
        #copyfile(FLAGS.data_cfg, FLAGS.log + "/semantic-kitti.yaml")
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting...")
        quit()

    trainer = Trainer(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.checkpoint, FLAGS.pretrained, FLAGS.fp16)
    trainer.train()