# data generator
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os
import numpy as np
import pandas as pd
import nibabel as nb
import torch
import matplotlib.pyplot as plt
sys.path.append('/host/d/Github')
import Whole_heart_segmentation_junzhe.functions_collection as ff
import Whole_heart_segmentation_junzhe.Data_processing as Data_processing
import Whole_heart_segmentation_junzhe.data_loader.random_aug as random_aug




# main function:
class Dataset_CMR(torch.utils.data.Dataset):
    def __init__(
            self, 
            image_file_list,
            seg_file_list,

            slice_num = 5,
            slice_range = None,


            shuffle = None,
            image_normalization = True,
            augment = None,
            augment_frequency = 0.5, # how often do we do augmentation
            rotate_range = [-10,10],
            translate_range = [-10,10],
            ):

        super().__init__()
        self.image_file_list = image_file_list
        self.seg_file_list = seg_file_list
      
        self.shuffle = shuffle
        self.image_normalization = image_normalization
        self.augment = augment
        self.augment_frequency = augment_frequency
        self.rotate_range = rotate_range
        self.translate_range = translate_range
        self.slice_num = slice_num
        self.slice_range = slice_range

        # how many cases we have in this dataset?
        self.num_files = len(self.image_file_list)

        # the following two should be run at the beginning of each epoch
        # 1. get index array
        self.index_array = self.generate_index_array()

        # 2. some parameters
        self.current_image_file = None
        self.current_image_data = None 
        self.current_seg_file = None
        self.current_seg_data = None

    # function: how many sample do we have in this dataset? 
    def __len__(self):
        return self.num_files
        
    # function: we need to generate an index array for dataloader, it's a list, each element is [file_index, slice_index]
    def generate_index_array(self):
        np.random.seed()
                
        if self.shuffle == True:
            file_index_list = np.random.permutation(self.num_files)
        else:
            file_index_list = np.arange(self.num_files)

        index_array = file_index_list.tolist()  # each element is file index now

        return index_array
    
    # function: 
    def load_file(self, filename, segmentation_load = False):
        ii = nb.load(filename).get_fdata()

        if segmentation_load is True:
            ii = np.round(ii).astype(int)
            ii_new = np.zeros_like(ii)
            ii_new[ii==500] = 1
            ii_new[ii==600] = 2
            ii_new[ii==420] = 3
            ii_new[ii==550] = 4
            ii_new[ii==205] = 5
            ii_new[ii==820] = 6
            ii_new[ii==850] = 7
            ii = np.copy(ii_new)

    
        return ii
    

    # function: get each item using the index [file_index]
    def __getitem__(self, index):
        f = self.index_array[index]
        image_filename = self.image_file_list[f]
        seg_filename = self.seg_file_list[f]
        print('loading image file:', image_filename, ' seg file:', seg_filename)

        # check if manual seg exists
        if os.path.isfile(seg_filename) is False:
            self.have_manual_seg = False
        else:
            self.have_manual_seg = True
            
        # if it's a new case, then do the data loading; if it's not, then just use the current data
        if image_filename != self.current_image_file or seg_filename != self.current_seg_file:
            image_loaded = self.load_file(image_filename, segmentation_load = False) 

            if self.have_manual_seg is True:
                seg_loaded = self.load_file(seg_filename, segmentation_load = True) 
            else:
                seg_loaded = np.zeros(image_loaded.shape, dtype = np.int)

        # crop or pad
        # 对于Unet来说，如果降采样了n次，那么输入的尺寸最好是2^n的倍数
        # 同时，sam的encoding是16的倍数
        # 所以我们的size_x和size_y都设置为(2^n*16)的倍数
        # 假设n=3，那么size_x和size_y都设置为128的倍数，选择有[128,256,384,512]
        size_candidtates = [128,256,384,512]
        size_x = image_loaded.shape[0]
        size_y = image_loaded.shape[1]
        assert size_x == size_y

        # 对于size_x来说，candidates中大于等于size_x的最小值是什么？
        target_size_x = XXXXXXX
        image_loaded = Data_processing.crop_or_pad(image_loaded, target_size = [target_size_x, target_size_x, image_loaded.shape[2]], padding_value = np.min(image_loaded))
        seg_loaded = Data_processing.crop_or_pad(seg_loaded, target_size = [target_size_x, target_size_x, seg_loaded.shape[2]], padding_value = np.min(seg_loaded))

        # 随机选slice_num个slice
        if self.slice_range is not None:
            slice_start = self.slice_range[0]
            slice_end = XXXXX
        else:
            slice_start随机选XXXXXXXX
            slice_end = slice_start + self.slice_num
        start_slice = np.random.randint(0, image_loaded.shape[2] - 5)
        image_loaded = image_loaded[:,:, start_slice : start_slice + 5]
        seg_loaded = seg_loaded[:,:, start_slice : start_slice + 5]


        # # center crop
        # if self.have_manual_seg is True:
        #     # find centroid based on the segmenation class 1
        #     _,_, self.centroid = Data_processing.center_crop( image_loaded, seg_loaded, self.image_shape, according_to_which_class = self.center_crop_according_to_which_class , centroid = None)

        # elif self.have_manual_seg is False:
        #     # center is the image center
        #     self.centroid = [image_loaded.shape[0]//2, image_loaded.shape[1]//2]

        #  # random crop (randomly shift the centroid)
        # if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
        #     random_centriod_shift_x = np.random.randint(-5,5)
        #     random_centriod_shift_y = np.random.randint(-5,5)
        #     centroid_used_for_crop = [self.centroid[0] + random_centriod_shift_x, self.centroid[1] + random_centriod_shift_y]
        # else:
        #     centroid_used_for_crop = self.centroid
                
        # # crop this 2D case
        # image_loaded = image_loaded[centroid_used_for_crop[0] - self.image_shape[0]//2 : centroid_used_for_crop[0] + self.image_shape[0]//2,
        #                                 centroid_used_for_crop[1] - self.image_shape[1]//2 : centroid_used_for_crop[1] + self.image_shape[1]//2 ]
        # seg_loaded = seg_loaded[centroid_used_for_crop[0] - self.image_shape[0]//2 : centroid_used_for_crop[0] + self.image_shape[0]//2,
        #                                 centroid_used_for_crop[1] - self.image_shape[1]//2 : centroid_used_for_crop[1] + self.image_shape[1]//2 ]
        
        # temporarily save our data
        self.current_image_file = image_filename
        self.current_image_data = np.copy(image_loaded)  
        self.current_seg_file = seg_filename
        self.current_seg_data = np.copy(seg_loaded)

        # augmentation
        original_image = np.copy(image_loaded)
        original_seg = np.copy(seg_loaded)
      
        ######## do augmentation
        processed_seg = np.copy(original_seg)
        # (0) add noise
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            standard_deviation = 5
            processed_image = original_image + np.random.normal(0,standard_deviation,original_image.shape)
            # turn the image pixel range to [0,255]
            processed_image = Data_processing.turn_image_range_into_0_255(processed_image)
        else:
            processed_image = Data_processing.turn_image_range_into_0_255(original_image)
       
        # (1) do brightness
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            processed_image,v = random_aug.random_brightness(processed_image, v = None)
    
        # (2) do contrast
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            processed_image, v = random_aug.random_contrast(processed_image, v = None)

        # (3) do sharpness
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            processed_image, v = random_aug.random_sharpness(processed_image, v = None)
            
        # (4) do flip
        # if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
        #     # doing this can make sure the flip is the same for image and seg
        #     a, selected_option = random_aug.random_flip(processed_image)
        #     b,_ = random_aug.random_flip(processed_seg, selected_option)
        #     processed_image = np.copy(a)
        #     processed_seg = np.copy(b)

        # (5) do rotate
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            processed_image, z_rotate_degree = random_aug.random_rotate(processed_image, order = 1, z_rotate_range = self.rotate_range)
            processed_seg,_ = random_aug.random_rotate(processed_seg, z_rotate_degree, fill_val = 0, order = 0)

        # (6) do translate
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            processed_image, x_translate, y_translate = random_aug.random_translate(processed_image, translate_range = self.translate_range)
            processed_seg,_ ,_= random_aug.random_translate(processed_seg, x_translate, y_translate)

        # add normalization
        if self.image_normalization is True:
            processed_image = Data_processing.normalize_image(processed_image,inverse = False) 

        image_size = processed_image.shape
        print('after augmentation, image min:', np.min(processed_image), ' max:', np.max(processed_image))

        # put into torch tensor
        processed_image = torch.from_numpy(processed_image).float().unsqueeze(0)  # add channel dimension
        processed_seg = torch.from_numpy(processed_seg).float().unsqueeze(0)  # add channel dimension
        original_image = torch.from_numpy(original_image).float().unsqueeze(0)  # add channel dimension
        original_seg = torch.from_numpy(original_seg).float().unsqueeze(0)  #

        # put into a dictionary
        
        final_dictionary = { "image": processed_image, 
                            "mask": processed_seg,
                            "image_size": image_size,
                            "original_image": original_image,  
                            "original_seg": original_seg,}


        return final_dictionary
          
    
    
    # function: at the end of each epoch, we need to reset the index array
    def on_epoch_end(self):
        print('now run on_epoch_end function')
        self.index_array = self.generate_index_array()

        self.current_image_file = None
        self.current_image_data = None 
        self.current_seg_file = None
        self.current_seg_data = None