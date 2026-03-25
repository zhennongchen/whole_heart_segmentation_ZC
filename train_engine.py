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
import torch.nn as nn
import torch.nn.functional as F
import whole_heart_segmentation_ZC.functions_collection as ff

## modify by ZC 03/25
class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, mask):
        """
        pred: [B, C, H, W]   raw logits
        mask: [B, H, W]   integer class labels
        """
        target = mask

        # ordinary pixel-wise cross entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none')   # [B, H, W]

        # pt = probability of the ground-truth class
        pt = torch.exp(-ce_loss)

        # focal cross entropy
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        


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

    ## modify by ZC 03/25
    if args.CE == 'focal':
        criterionBCE = FocalCrossEntropyLoss(gamma=2.0)
    else:
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
            
            # focal CE
            lossCE = criterionBCE(output["masks"], torch.clone(mask).squeeze(1).long()) 
            # DICE loss, ## modify by ZC 03/25
            lossDICE = ff.customized_dice_loss(output["masks"], torch.clone(mask).squeeze(1).long(), num_classes = args.num_classes, only_present_mask = args.DICE_only_present_mask)#, exclude_index = args.turn_zero_seg_slice_into)
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



def validation_loop(model, data_loader_valid,  args): 
        
    with torch.no_grad():
                                    
        # model.eval()
        if args.CE == 'focal':
            criterionBCE = FocalCrossEntropyLoss(gamma=2.0)
        else:
            if args.turn_zero_seg_slice_into is not None:
                criterionBCE = torch.nn.CrossEntropyLoss(ignore_index=10)
                # print('in train loop we have turn_zero_seg_slice_into: ', args.turn_zero_seg_slice_into)
            else:
                criterionBCE = torch.nn.CrossEntropyLoss()

        average_valid_loss = []
        average_valid_lossCE = []
        average_valid_lossDICE = []

        for data_iter_step, batch in tqdm(enumerate(data_loader_valid)):
            # Note that our input shape 
            batch["image"]= batch["image"].cuda()
            # print('in prediction batch image shape: ', batch["image"].shape)

            output = model(batch, args.img_size)

            mask = batch["mask"]
            # print('in train engine batch mask shape initially: ', mask.shape)
            mask = rearrange(mask, 'b c h w d -> (b d) c h w ').to("cuda")

            #### CE loss
            lossCE = criterionBCE(output["masks"], mask.squeeze(1).long()) 
            
            #### customized dice loss
            mask_for_dice = batch["mask"]
            mask_for_dice = rearrange(mask_for_dice, 'b c h w d -> (b d) c h w').to("cuda")
            lossDICE = ff.customized_dice_loss(output["masks"], mask_for_dice.squeeze(1).long(), num_classes = args.num_classes, exclude_index = args.turn_zero_seg_slice_into, only_present_mask = args.DICE_only_present_mask)

            #### total loss: weighted loss
            loss = args.loss_weights[0] * lossCE + args.loss_weights[1] * lossDICE
            
            if torch.isnan(loss):
                continue
            
            average_valid_loss.append(loss.item())
            average_valid_lossCE.append(lossCE.item())
            average_valid_lossDICE.append(lossDICE.item())

            torch.cuda.synchronize()

        average_valid_loss = sum(average_valid_loss)/len(average_valid_loss)
        average_valid_lossCE = sum(average_valid_lossCE)/len(average_valid_lossCE)
        average_valid_lossDICE = sum(average_valid_lossDICE)/len(average_valid_lossDICE)
       
    return average_valid_loss, average_valid_lossCE, average_valid_lossDICE