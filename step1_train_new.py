import os
import sys
sys.path.append('/host/d/Github')  ### remove this if not needed!
import numpy as np
import pandas as pd 
from tqdm import tqdm 
from pathlib import Path
import nibabel as nb
import matplotlib.pyplot as plt

import argparse
from einops import rearrange 
from natsort import natsorted
from madgrad import MADGRAD

import torch
import torch.backends.cudnn as cudnn

from whole_heart_segmentation_ZC.utils.model_util import *
from whole_heart_segmentation_ZC.segment_anything.model import build_model 
from whole_heart_segmentation_ZC.utils.save_utils import *
from whole_heart_segmentation_ZC.utils.config_util import Config
from whole_heart_segmentation_ZC.utils.misc import NativeScalerWithGradNormCount as NativeScaler

from whole_heart_segmentation_ZC.train_engine import train_loop

import whole_heart_segmentation_ZC.functions_collection as ff
import whole_heart_segmentation_ZC.Data_processing as Data_processing
import whole_heart_segmentation_ZC.Build_lists.Build_list as Build_list
import whole_heart_segmentation_ZC.data_loader.generator as generator
from torch.utils.data import Dataset, DataLoader


main_path = '/host/d/projects/WHS/' ### change to your main path



### step 1: define trial name
trial_name = 'trial_WHS'
output_dir = os.path.join(main_path, 'models', trial_name) # change to your output dir
ff.make_folder([output_dir])

### step 2: define parameters for this trial
# many important parameters, focus on ones that I comment with ###!!

def get_args_parser(img_size = 512, num_classes = 3, slice_num = 5, pretrained_model = None, original_sam = None, start_epoch = None, total_training_epochs = 1000, save_model_every = 1,  vit_type = "vit_b"):
    parser = argparse.ArgumentParser('SAM fine-tuning', add_help=True)

    # img size
    parser.add_argument('--img_size', default=img_size, type=int)  ## !!

    ## augmentation
    parser.add_argument('--augment_frequency', default= 0.5, type=float) ## !! ise a proper frequency

    ## segmentation classes
    parser.add_argument('--num_classes', type=int, default=num_classes) ## !!

    ## pretrained sam
    parser.add_argument('--resume', default = original_sam) ##!!

    # for training
    parser.add_argument('--total_training_epochs', default = total_training_epochs, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)') ##!!
    parser.add_argument('--print_freq', default=10, type = int) 
    parser.add_argument('--save_model_file_every_N_epoch', default=save_model_every, type = int)  ## !!
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')  ## !!
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256') ## !!
    parser.add_argument('--lr_update_every_N_epoch', default=100, type = int) ## !!
    parser.add_argument('--lr_decay_gamma', default=0.95)
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--loss_weights', default = [1,1] )  #### !! weighting for loss function [BCE, Dice]

    if start_epoch == None:
        parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='start epoch')
    else:
        parser.add_argument('--start_epoch', default= start_epoch, type=int, metavar='N', help='start epoch')

    # standard
    parser.add_argument('--text_prompt', default = False)
    parser.add_argument('--box_prompt', default= False) 
    parser.add_argument('--pretrained_model', default = pretrained_model)
    
    parser.add_argument('--validation', default=False) ## !!
    parser.add_argument('--cross_frame_attention', default=False) # False

    parser.add_argument('--model_type', type=str, default='sam')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu') 
    parser.add_argument('--use_amp', action='store_true', help='If activated, adopt mixed precision for acceleration')
    parser.add_argument("--config", help="Path to the training config file.", default="configs/config.yaml",)

    parser.add_argument('--seed', default=1234, type=int)   
    parser.add_argument('--input_type', type=str, default='2DT') #has to be 2DT
    parser.add_argument('--vit_type', type=str, default=vit_type)
    parser.add_argument('--slice_num', default=slice_num, type=int) 
                        

    parser.add_argument('--turn_zero_seg_slice_into', default=10, type=int)
 
    return parser


pretrained_model = None#os.path.join(output_dir, 'model-1000.pth')
start_epoch = 1 # 1 if no pretrained model
total_training_epochs = 200 # change to a reasonable number

# define the original sam model weights (you should download it from online to your local path)
original_sam = '/host/d/Data/pretrained_SAM_weights/sam_vit_b.pth'  # change to your original sam model path

# pick how many consecutive slices to construct a 3D volume
slice_num = 5

args = get_args_parser(img_size = 512, ## important !! need to change based on your dataset
        num_classes = 8, ## important !! need to change based on your dataset
        slice_num = slice_num,
        
        pretrained_model = pretrained_model, 
        original_sam = original_sam, 
        start_epoch = start_epoch, 
        total_training_epochs = total_training_epochs, 
        save_model_every = 1,
        vit_type = "vit_b",)

args = args.parse_args([])

# some other settings
cfg = Config(args.config)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True



### step 3: build patient list
# # change the excel path to your own path
patient_list_spreadsheet = os.path.join('/host/d/Data/WHS/Patient_lists','train_val_path_list.xlsx')
build_sheet =  Build_list.Build(patient_list_spreadsheet)
# train
_, _, _, _, size_x_list_train, size_y_list_train, size_z_list_train, img_file_list_train, seg_file_list_train = build_sheet.__build__(batch_list = [0,1,2])  # just as an example, use batch 0 for train

# define val
_, _, _, _, size_x_list_val, size_y_list_val, size_z_list_val, img_file_list_val, seg_file_list_val = build_sheet.__build__(batch_list = [3])  # just as an example, use batch 3 for val

print('train ', len(img_file_list_train))
print('val ', len(img_file_list_val))



### step 4: load pre-trained SAM model (freeze SAM blocks)
# set model
model = build_model(args, device)

# set freezed and trainable keys
train_keys = []
freezed_keys = []
        
# load pretrained sam model vit_h
if args.model_type.startswith("sam"):
    if args.resume.endswith(".pth"):
        with open(args.resume, "rb") as f:
            state_dict = torch.load(f)
        try:
            model.load_state_dict(state_dict)
        except:
            if args.vit_type == "vit_h" or args.vit_type == "vit_l" or args.vit_type == "vit_b":
                new_state_dict = load_from(model, state_dict, args.img_size,  16, [7, 15, 23, 31])
               
            model.load_state_dict(new_state_dict)
        
        # freeze original SAM layers
        freeze_list = [ "norm1", "attn" , "mlp", "norm2"]  
                
        for n, value in model.named_parameters():
            if any(substring in n for substring in freeze_list):
                freezed_keys.append(n)
                value.requires_grad = False
            else:
                train_keys.append(n)
                value.requires_grad = True

## Select optimization method
optimizer = MADGRAD(model.parameters(), lr=args.lr)
        
# Continue training model
if args.pretrained_model is not None:
    if os.path.exists(args.pretrained_model):
        print('loading pretrained model : ', args.pretrained_model)
        args.resume = args.pretrained_model
        finetune_checkpoint = torch.load(args.pretrained_model)
        model.load_state_dict(finetune_checkpoint["model"])
        optimizer.load_state_dict(finetune_checkpoint["optimizer"])
        torch.cuda.empty_cache()
else:
    print('new training\n')



### final step: let's train!!
# define data generator
generator_train = generator.Dataset_CMR(
            image_file_list = img_file_list_train,
            seg_file_list = seg_file_list_train,
        
            args = args,
            how_many_slices_set_per_case = 5,
            slice_range = None,

            shuffle = True,
            image_normalization = True,
            augment = True,
            augment_frequency = args.augment_frequency )


# training loader
data_loader_train = DataLoader(generator_train, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())

training_log = []
model_save_folder = os.path.join(output_dir, 'models'); ff.make_folder([output_dir, model_save_folder])
log_save_folder = os.path.join(output_dir, 'logs'); ff.make_folder([log_save_folder])

for epoch in range(args.start_epoch,  args.total_training_epochs+1):
        print('training epoch:', epoch)

        if epoch % args.lr_update_every_N_epoch == 0:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * args.lr_decay_gamma
        # print('learning rate now:', optimizer.param_groups[0]["lr"])
        
        loss_scaler = NativeScaler()
            
        train_results = train_loop(
                model = model,
                data_loader_train  = data_loader_train,
                optimizer = optimizer,
                epoch = epoch, 
                loss_scaler = loss_scaler,
                args = args,
                inputtype = cfg.data.input_type)   
              
        loss, lossCE, lossDICE = train_results
        print('in epoch: ', epoch, ' training average_loss: ', loss, ' average_lossCE: ', lossCE, ' average_lossDICE: ', lossDICE,)
    
        # on_epoch_end:
        generator_train.on_epoch_end()
    
        if  epoch % args.save_model_file_every_N_epoch == 0 or epoch  == args.total_training_epochs:
            checkpoint_path = os.path.join(model_save_folder,  'model-%s.pth' % epoch)
            to_save = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,}
            torch.save(to_save, checkpoint_path)

        training_log.append([epoch, optimizer.param_groups[0]["lr"], loss, lossCE, lossDICE])
        df = pd.DataFrame(training_log, columns=['epoch', 'lr','training_loss', 'training_lossCE', 'training_lossDICE'])
        df.to_excel(os.path.join(log_save_folder, 'training_log.xlsx'), index=False)