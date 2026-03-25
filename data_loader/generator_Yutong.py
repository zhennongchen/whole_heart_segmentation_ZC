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
import whole_heart_segmentation_ZC.functions_collection as ff
import whole_heart_segmentation_ZC.Data_processing as Data_processing
import whole_heart_segmentation_ZC.data_loader.random_aug as random_aug

def split_slice_range(slice_range, slice_num, how_many_slices_set_per_case):
    a, b = slice_range
    assert (b - a) == slice_num * how_many_slices_set_per_case, \
        "Range length does not match slice_num * how_many_slices_set_per_case"

    sections = []
    for i in range(how_many_slices_set_per_case):
        start = a + i * slice_num
        end = start + slice_num
        sections.append([start, end])

    return sections


# main function:
class Dataset_CMR_Simple(torch.utils.data.Dataset):
    def __init__(
            self, 
            image_file_list,
            seg_file_list,

            args = None,
            how_many_slices_set_per_case = None,
            slice_range = None,
            ratio_rich_foreground = 0.8,

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
        self.args = args

        self.how_many_slices_set_per_case = how_many_slices_set_per_case
        self.slice_range = slice_range
        self.slice_num = self.args.slice_num
        self.ratio_rich_foreground = ratio_rich_foreground
      
        self.shuffle = shuffle
        self.image_normalization = image_normalization
        self.augment = augment
        self.augment_frequency = augment_frequency
        self.rotate_range = rotate_range
        self.translate_range = translate_range
        
        if self.slice_range is not None:
            assert (self.slice_range[1]- self.slice_range[0]) == self.slice_num * self.how_many_slices_set_per_case, "the total number of slices should match the product of slice_num and how_many_slices_set_per_case"
        
        # how many samples we have
        self.num_samples = len(self.image_file_list) * self.how_many_slices_set_per_case

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
        return self.num_samples
        
    # function: we need to generate an index array for dataloader, it's a list, each element is [file_index, slice_index]
    def generate_index_array(self):
        np.random.seed()
        index_array = []

        if self.shuffle == True:
            file_index_list = np.random.permutation(len(self.image_file_list))
        else:
            file_index_list = np.arange(len(self.image_file_list))

        for f in file_index_list:
            set_index_list = np.arange(self.how_many_slices_set_per_case)
            for s in set_index_list:
                index_array.append([f, s])
                
        return index_array

    
    # function: 
    def load_file(self, filename, segmentation_load = False):
        ii = nb.load(filename).get_fdata()

        if segmentation_load is True:
            ii = np.round(ii).astype(int)
            ii_new = np.zeros_like(ii)
            ii_new[ii==500] = 1
            ii_new[ii==600] = 2
            # ii_new[ii==420] = 3
            # ii_new[ii==550] = 4
            ii_new[ii==205] = 3
            # ii_new[ii==820] = 6
            # ii_new[ii==850] = 7

            # according to args.num_classes, only keep these classes
            # ii_new[ii_new >= self.args.num_classes] = 0
            
            ii = np.copy(ii_new)
            # print('unique classes in segmentation:', np.unique(ii))
    
        return ii
    

    # function: get each item using the index [file_index]
    def __getitem__(self, index):
        f,s = self.index_array[index]
        image_filename = self.image_file_list[f]
        seg_filename = self.seg_file_list[f]
        

        # check if manual seg exists
        if os.path.isfile(seg_filename) is False:
            self.have_manual_seg = False
        else:
            self.have_manual_seg = True

            
        # if it's a new case, then do the data loading; if it's not, then just use the current data
        if image_filename != self.current_image_file or seg_filename != self.current_seg_file:
            # print('loading image file:', image_filename, ' seg file:', seg_filename)
            image_loaded = self.load_file(image_filename, segmentation_load = False) 

            if self.have_manual_seg is True:
                seg_loaded = self.load_file(seg_filename, segmentation_load = True) 
            else:
                seg_loaded = np.zeros(image_loaded.shape, dtype = np.int)
            
            self.current_image_data = image_loaded
            self.current_seg_data = seg_loaded
            self.current_image_file = image_filename 
            self.current_seg_file = seg_filename

            # set the slice sets
            # in each set, we have args.slice_num个slices, 
            if self.slice_range is not None:
                self.slices_set = split_slice_range(self.slice_range, self.slice_num, self.how_many_slices_set_per_case)
            else:
                max_start = image_loaded.shape[2] - self.slice_num

                # first we need to the percentage of foreground pixels in each slices
                self.foreground_percentage = []
                for ss in range(0,max_start+1):
                    slice_seg = seg_loaded[:,:,ss]
                    foreground_pixels = np.sum(slice_seg > 0)
                    total_pixels = slice_seg.size
                    self.foreground_percentage.append(foreground_pixels / total_pixels)
                    # print('slice ss:', ss, 'foreground percentage:', self.foreground_percentage[-1])
                self.mean_foreground_percentage = np.mean(self.foreground_percentage)

                # then, found out slices with percentage larger than self.mean_foreground_percentage, and slices smaller
                self.rich_slices = [i for i, p in enumerate(self.foreground_percentage) if p > self.mean_foreground_percentage]
                self.scare_slices = [i for i, p in enumerate(self.foreground_percentage) if p <= self.mean_foreground_percentage]

                # totally randomly pick
                if self.ratio_rich_foreground is None:
                    starts = np.random.randint(0, max_start + 1, size=self.how_many_slices_set_per_case)
                else:
                    num_rich_slices = int(self.ratio_rich_foreground * self.how_many_slices_set_per_case)
                    num_scare_slices = self.how_many_slices_set_per_case - num_rich_slices
                    rich_starts = np.random.choice(self.rich_slices, size=num_rich_slices, replace=False)
                    scare_starts = np.random.choice(self.scare_slices, size=num_scare_slices, replace=False)
                    starts = np.concatenate([rich_starts, scare_starts])

                self.slices_set = [[int(s), int(s + self.slice_num)] for s in starts]
 
            # print('slice sets is ', self.slices_set)
        else:
            image_loaded = self.current_image_data
            seg_loaded = self.current_seg_data

        # crop or pad
        target_size_x = self.args.img_size 
        image_loaded = Data_processing.crop_or_pad(image_loaded, target_size = [target_size_x, target_size_x, image_loaded.shape[2]], padding_value = np.min(image_loaded))
        seg_loaded = Data_processing.crop_or_pad(seg_loaded, target_size = [target_size_x, target_size_x, seg_loaded.shape[2]], padding_value = np.min(seg_loaded))

        # pick index s in slices_set
        [start_slice, end_slice] = self.slices_set[s]
        # print('selected slice range:', start_slice, end_slice)
        image_loaded = image_loaded[:,:, start_slice : end_slice]
        seg_loaded = seg_loaded[:,:, start_slice : end_slice]


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
        # print('after augmentation, image min:', np.min(processed_image), ' max:', np.max(processed_image))

        # put into torch tensor
        processed_image = torch.from_numpy(processed_image).float().unsqueeze(0)  # add channel dimension
        processed_seg = torch.from_numpy(processed_seg).float().unsqueeze(0)  # add channel dimension
        original_image = torch.from_numpy(original_image).float().unsqueeze(0)  # add channel dimension
        original_seg = torch.from_numpy(original_seg).float().unsqueeze(0)  #
        # print('in original seg, the unique values are:', torch.unique(original_seg))


        # put into a dictionary
        
        final_dictionary = { "image": processed_image, 
                            "mask": processed_seg,
                            "image_size": image_size,
                            "original_image": original_image,  
                            "original_seg": original_seg,
                            "image_filename": self.current_image_file,
                            "slice_range": [start_slice, end_slice]}


        return final_dictionary
          
    
    
    # function: at the end of each epoch, we need to reset the index array
    def on_epoch_end(self):
        print('now run on_epoch_end function')
        self.index_array = self.generate_index_array()

        self.current_image_file = None
        self.current_image_data = None 
        self.current_seg_file = None
        self.current_seg_data = None





#### for sliding window
def generate_sliding_windows_and_counts_from_range(slice_range, slice_num=10, stride=1):
    """
    slice_range: [a, b], left-inclusive, right-exclusive
    valid slices are a, a+1, ..., b-1

    return:
        windows: list of [start, end]
        counts:  list of length (b-a), counting coverage within this range
                 counts[i] corresponds to global slice (a + i)
    """
    a, b = slice_range
    assert b > a, "slice_range should satisfy b > a"
    assert slice_num <= (b - a), "slice_num must be <= length of slice range"

    windows = []
    counts = [0] * (b - a)

    for start in range(a, b - slice_num + 1, stride):
        end = start + slice_num
        windows.append([start, end])

        for z in range(start, end):
            counts[z - a] += 1   # shift to local index

    return windows, counts


class Dataset_CMR_Simple_sliding(torch.utils.data.Dataset):
    def __init__(
            self, 
            image_file_list,
            seg_file_list,

            args = None,
   
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
        self.args = args

        self.slice_range = slice_range
        self.slice_num = self.args.slice_num
      
        self.shuffle = shuffle
        self.image_normalization = image_normalization
        self.augment = augment
        self.augment_frequency = augment_frequency
        self.rotate_range = rotate_range
        self.translate_range = translate_range
        
        # generate sliding windows
        self.windows, self.counts = generate_sliding_windows_and_counts_from_range(self.slice_range, self.slice_num, stride=1)
        print('windows:', self.windows)

        # how many samples we have
        self.num_samples = len(self.image_file_list) * len(self.windows)

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
        return self.num_samples
        
    # function: we need to generate an index array for dataloader, it's a list, each element is [file_index, slice_index]
    def generate_index_array(self):
        np.random.seed()
        index_array = []

        if self.shuffle == True:
            file_index_list = np.random.permutation(len(self.image_file_list))
        else:
            file_index_list = np.arange(len(self.image_file_list))

        for f in file_index_list:
            set_index_list = np.arange(len(self.windows))
            for s in set_index_list:
                index_array.append([f, s])
                
        return index_array

    
    # function: 
    def load_file(self, filename, segmentation_load = False):
        ii = nb.load(filename).get_fdata()

        if segmentation_load is True:
            ii = np.round(ii).astype(int)
            ii_new = np.zeros_like(ii)
            ii_new[ii==500] = 1
            ii_new[ii==600] = 2
            # ii_new[ii==420] = 3
            # ii_new[ii==550] = 4
            ii_new[ii==205] = 3
            # ii_new[ii==820] = 6
            # ii_new[ii==850] = 7

            # according to args.num_classes, only keep these classes
            # ii_new[ii_new >= self.args.num_classes] = 0
            
            ii = np.copy(ii_new)
            # print('unique classes in segmentation:', np.unique(ii))
    
        return ii
    

    # function: get each item using the index [file_index]
    def __getitem__(self, index):
        f,s = self.index_array[index]
        image_filename = self.image_file_list[f]
        seg_filename = self.seg_file_list[f]
        

        # check if manual seg exists
        if os.path.isfile(seg_filename) is False:
            self.have_manual_seg = False
        else:
            self.have_manual_seg = True

            
        # if it's a new case, then do the data loading; if it's not, then just use the current data
        if image_filename != self.current_image_file or seg_filename != self.current_seg_file:
            # print('loading image file:', image_filename, ' seg file:', seg_filename)
            image_loaded = self.load_file(image_filename, segmentation_load = False) 

            if self.have_manual_seg is True:
                seg_loaded = self.load_file(seg_filename, segmentation_load = True) 
            else:
                seg_loaded = np.zeros(image_loaded.shape, dtype = np.int)
            
            self.current_image_data = image_loaded
            self.current_seg_data = seg_loaded
            self.current_image_file = image_filename 
            self.current_seg_file = seg_filename

            # set the slice sets
            # sliding windows
            self.slices_set = self.windows
            # print('slice set:', self.slices_set)
 
            # print('slice sets is ', self.slices_set)
        else:
            image_loaded = self.current_image_data
            seg_loaded = self.current_seg_data

        # crop or pad
        target_size_x = self.args.img_size 
        image_loaded = Data_processing.crop_or_pad(image_loaded, target_size = [target_size_x, target_size_x, image_loaded.shape[2]], padding_value = np.min(image_loaded))
        seg_loaded = Data_processing.crop_or_pad(seg_loaded, target_size = [target_size_x, target_size_x, seg_loaded.shape[2]], padding_value = np.min(seg_loaded))

        # pick index s in slices_set
        [start_slice, end_slice] = self.slices_set[s]
        # print('selected slice range:', start_slice, end_slice)
        image_loaded = image_loaded[:,:, start_slice : end_slice]
        seg_loaded = seg_loaded[:,:, start_slice : end_slice]


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
        # print('after augmentation, image min:', np.min(processed_image), ' max:', np.max(processed_image))

        # put into torch tensor
        processed_image = torch.from_numpy(processed_image).float().unsqueeze(0)  # add channel dimension
        processed_seg = torch.from_numpy(processed_seg).float().unsqueeze(0)  # add channel dimension
        original_image = torch.from_numpy(original_image).float().unsqueeze(0)  # add channel dimension
        original_seg = torch.from_numpy(original_seg).float().unsqueeze(0)  #
        # print('in original seg, the unique values are:', torch.unique(original_seg))


        # put into a dictionary
        
        final_dictionary = { "image": processed_image, 
                            "mask": processed_seg,
                            "image_size": image_size,
                            "original_image": original_image,  
                            "original_seg": original_seg,
                            "image_filename": self.current_image_file,
                            "slice_range": [start_slice, end_slice],
                            "counts": self.counts,
                            "windows": self.windows}


        return final_dictionary
          
    
    
    # function: at the end of each epoch, we need to reset the index array
    def on_epoch_end(self):
        print('now run on_epoch_end function')
        self.index_array = self.generate_index_array()

        self.current_image_file = None
        self.current_image_data = None 
        self.current_seg_file = None
        self.current_seg_data = None