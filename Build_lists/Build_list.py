import numpy as np
import os
import pandas as pd


class Build():
    def __init__(self,file_list):
        self.file_list = file_list
        self.data = pd.read_excel(file_list, dtype = {'patient_id': str})

    def __build__(self,batch_list):
        for b in range(len(batch_list)):
            cases = self.data.loc[self.data['batch'] == batch_list[b]]
            if b == 0:
                c = cases.copy()
            else:
                c = pd.concat([c,cases])

        batch_list = np.asarray(c['batch'])
        modality_list = np.asarray(c['modality'])
        center_list = np.asarray(c['center'])
        patient_id_list = np.asarray(c['patient_id'])
        size_x_list = np.asarray(c['size_x'])
        size_y_list = np.asarray(c['size_y'])
        size_z_list = np.asarray(c['size_z'])
        
        img_file_list = np.asarray(c['img_path'])
        seg_file_list = np.asarray(c['label_path'])
        
        return batch_list, modality_list, center_list, patient_id_list, size_x_list, size_y_list, size_z_list, img_file_list, seg_file_list