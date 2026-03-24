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
import whole_heart_segmentation_ZC.functions_collection as ff


def save_predictions(batch, output,args, save_folder_patient):
    pred_seg = np.rollaxis(output["masks"].argmax(1).detach().cpu().numpy(), 0, 3)
    # if pred_seg has shape (H,W,1), we squeeze it to (H,W)
    pred_seg = np.squeeze(pred_seg)

    # we need to turn pred_seg back to the original image size
    original_image_file = batch["image_filename"][0]        

    affine = nb.load(original_image_file).affine
    original_image = nb.load(original_image_file).get_fdata()
    original_shape = original_image.shape

    # original_shape = np.array([x.item() for x in batch["original_shape"]])
    # if original shape dim = 3 and the last dim is 1, we squeeze it to 2D
    if len(original_shape) ==3 and original_shape[2]==1:
        original_shape = original_shape[:2]
    centroid = batch["centroid"].numpy().flatten()
    print('original shape: ', original_shape)
              
    crop_start_end_list = []
    for dim, size in enumerate([args.img_size, args.img_size]):
        start = max(centroid[dim] - size // 2, 0)
        end = start + size
        # Adjust the start and end if they are out of bounds
        if end > original_shape[dim]:
            end = original_shape[dim]
            start = max(end - size, 0)
        crop_start_end_list.append([start, end])
     
    final_pred_seg = np.zeros(original_shape)
    final_pred_seg[crop_start_end_list[0][0]:crop_start_end_list[0][1], crop_start_end_list[1][0]:crop_start_end_list[1][1]] = pred_seg

    

    nb.save(nb.Nifti1Image(final_pred_seg, affine), os.path.join(save_folder_patient, 'pred_seg.nii.gz'))
    nb.save(nb.Nifti1Image(original_image, affine), os.path.join(save_folder_patient, 'img.nii.gz'))





