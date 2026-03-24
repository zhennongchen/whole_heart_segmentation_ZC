import os
from tqdm import tqdm
import torch
import nibabel as nb
import numpy as np
from typing import Iterable
# from tensorboardX import SummaryWriter
import logging
from einops import rearrange
import utils.misc as misc
import utils.lr_sched as lr_sched
import torch.nn.functional as F
import whole_heart_segmentation_ZC.functions_collection as ff

def train_loop(model: torch.nn.Module,
               data_loader_train: Iterable,
               optimizer: torch.optim.Optimizer,
               epoch: int, 
               loss_scaler,
               args=None,
               inputtype = None):
    
    # # make some settings
    # metric_logger = misc.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # header = 'Epoch: [{}]'.format(epoch)
     
    accum_iter = args.accum_iter
    
    model.train(True)
    
    if args.turn_zero_seg_slice_into is not None:
        criterionBCE = torch.nn.CrossEntropyLoss(ignore_index=10)
        # print('in train loop we have turn_zero_seg_slice_into: ', args.turn_zero_seg_slice_into)
    else:
        criterionBCE = torch.nn.CrossEntropyLoss()

    # start to train
    average_loss = []; average_lossCE = []; average_lossDICE = []

        
    for data_iter_step, batch in enumerate(tqdm(data_loader_train)):
        with torch.cuda.amp.autocast():
            if (data_iter_step + 1) % accum_iter == 0 or data_iter_step == len(data_loader_train) - 1:
                optimizer.zero_grad()

            batch["image"]= batch["image"].float().cuda()
            # print('batch image shape:', batch["image"].shape)
                
            output = model(batch, args.img_size)
            # print('output shape: ',output["masks"].shape)

            mask = batch["mask"]
            mask = rearrange(mask, 'b c h w d -> (b d) c h w ').to("cuda")
            # print('mask shape: ', mask.shape, ' unique mask values: ', torch.unique(mask))
                   
            lossCE = criterionBCE(output["masks"], torch.clone(mask).squeeze(1).long()) 
            lossDICE = ff.customized_dice_loss(output["masks"], torch.clone(mask).squeeze(1).long(), num_classes = args.num_classes)#, exclude_index = args.turn_zero_seg_slice_into)
            # print('lossCE:', lossCE.item(), ' lossDICE:', lossDICE.item())

            #### total loss: weighted loss
            loss = args.loss_weights[0] * lossCE + args.loss_weights[1] * lossDICE
                   
            if torch.isnan(loss):
                continue

            subset_params = [p for p in model.parameters()]
            # print('number of subset params:', len(subset_params))
                
            # I'll not use the complicated loss scaler here
            # loss_scaler(loss, optimizer, parameters=subset_params,update_grad = True)#, update_grad=(data_iter_step + 1) % accum_iter == 0)  
            # backward using the easiest way
            if (data_iter_step + 1) % accum_iter == 0 or data_iter_step == len(data_loader_train) - 1:
                loss.backward()
                optimizer.step()
          
            torch.cuda.synchronize()
            # metric_logger.update(loss1=loss.item())
            # lr = optimizer.param_groups[0]["lr"]
            # metric_logger.update(lr=lr)

            # metric_logger.synchronize_between_processes()
            # print("Averaged stats:", metric_logger)
        
            average_loss.append(loss.item()); average_lossCE.append(lossCE.item()); average_lossDICE.append(lossDICE.item())
            

    average_loss_mean = sum(average_loss)/len(average_loss);average_lossCE_mean = sum(average_lossCE)/len(average_lossCE);average_lossDICE_mean = sum(average_lossDICE)/len(average_lossDICE)
    
    
    return [average_loss_mean, average_lossCE_mean, average_lossDICE_mean]
