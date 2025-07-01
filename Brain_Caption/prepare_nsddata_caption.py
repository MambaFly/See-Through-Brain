'''
根据下载的nsd数据，准备brain_caption模型所需的训练集和测试集

主要功能:
1. 加载并处理fMRI数据
2. 加载并处理视觉刺激数据
3. 加载并处理字幕数据
4. 准备训练集和测试集
5. 保存处理后的数据
'''

### 导入必要的库
# 基础库
import os
import sys
from os.path import join as opj
# 数学计算和数据处理库
import numpy as np
import pandas as pd
# 文件读取库
import h5py
import scipy.io as spio
import nibabel as nib
# 辅助库
import json
import tqdm
import argparse

class NSDDataProcessor:
    def __init__(self, subject='subj07', subj=7, base_path="./"):
        """初始化数据处理器"""
        self.subject = subject
        self.subj = subj
        self.base_path = base_path
        self.setup_paths()
        self.setup_directories()
        
    def setup_paths(self):
        """设置所有数据文件路径"""
        # 基础路径设置
        self.betas_path = opj(self.base_path, "nsddata_betas")
        self.stim_info_path = opj(self.base_path, "nsddata", "experiments", "nsd", "nsd_stim_info_merged.csv")
        
        # 刺激数据路径
        self.stimuli_path = opj(self.base_path, "nsddata_stimuli", "stimuli", "nsd")
        self.stim_file_path = opj(self.stimuli_path, "nsd_stimuli.hdf5")
        
        # 字幕数据路径
        self.stim_captions_train_path = opj(self.stimuli_path, "annotations", "captions_train2017.json")
        self.stim_captions_val_path = opj(self.stimuli_path, "annotations", "captions_val2017.json")
        
        # ROI和betas路径
        self.roi_dir = f'nsddata/ppdata/subj{self.subj:02d}/func1pt8mm/roi/'
        self.betas_dir = f'nsddata_betas/ppdata/subj{self.subj:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'

    def setup_directories(self):
        """创建保存处理后数据的目录"""
        self.savedir = f'./processed_data/{self.subject}/'
        os.makedirs(self.savedir, exist_ok=True)

    def load_stimulus_info(self):
        """加载刺激信息和字幕数据"""
        self.stim_info = pd.read_csv(self.stim_info_path)
        
        # 加载字幕数据
        with open(self.stim_captions_train_path, 'rb') as f:
            train_cap = json.load(f)
        with open(self.stim_captions_val_path, 'rb') as f:
            val_cap = json.load(f)
            
        self.caption_train_df = pd.DataFrame.from_dict(train_cap["annotations"])
        self.caption_val_df = pd.DataFrame.from_dict(val_cap["annotations"])

    def prepare_trial_indices(self):
        """准备训练和测试数据的试验索引"""
        # 加载实验设计数据
        stim_order = self.loadmat('nsddata/experiments/nsd/nsd_expdesign.mat')
        
        self.sig_train = {}
        self.sig_test = {}
        num_trials = 40 * 750
        
        # 分配训练和测试索引
        for idx in range(num_trials):
            nsdId = stim_order['subjectim'][self.subj-1, stim_order['masterordering'][idx] - 1] - 1
            # 训练数据
            if stim_order['masterordering'][idx] > 1000:
                if nsdId not in self.sig_train:
                    self.sig_train[nsdId] = []
                self.sig_train[nsdId].append(idx)
            # 测试数据
            else:
                if nsdId not in self.sig_test:
                    self.sig_test[nsdId] = []
                self.sig_test[nsdId].append(idx)

        # 训练数据和测试数据对应的刺激索引
        self.train_im_idx = list(self.sig_train.keys())
        self.test_im_idx = list(self.sig_test.keys())
        print(len(self.train_im_idx))
        print(len(self.test_im_idx))

    def load_fmri_data(self):
        """加载fMRI数据"""
        # 加载ROI掩码
        mask = nib.load(self.roi_dir + 'nsdgeneral.nii.gz').get_fdata()
        self.num_voxel = mask[mask>0].shape[0]
        
        # 加载beta值
        self.fmri = np.zeros((40*750, self.num_voxel)).astype(np.float32)
        for i in range(40):
            beta_filename = f"betas_session{i+1:02d}.nii.gz"
            beta_f = nib.load(self.betas_dir + beta_filename).get_fdata().astype(np.float32)
            self.fmri[i*750:(i+1)*750] = beta_f[mask>0].transpose()
            print(f"Loaded session {i+1}/40")
        
        print("fMRI data are loaded.")

    def load_stimulus_data(self):
        """加载视觉刺激数据"""
        with h5py.File(self.stim_file_path, 'r') as f_stim:
            self.stim = f_stim['imgBrick'][:]
        print("Stimuli are loaded.")

    def save_training_data(self):
        """保存训练数据"""
        num_train = len(self.train_im_idx)
        fmri_array = np.zeros((num_train, self.num_voxel))
        stim_array = np.zeros((num_train, 425, 425, 3))
        
        for i, idx in enumerate(self.train_im_idx):
            stim_array[i] = self.stim[idx]
            fmri_array[i] = self.fmri[sorted(self.sig_train[idx])].mean(0)
            
        np.save(f'{self.savedir}/nsd_train_fmriavg_nsdgeneral_{self.subject}.npy', fmri_array)
        np.save(f'{self.savedir}/nsd_train_stim_{self.subject}.npy', stim_array)
        print("Training data are saved.")

    def save_test_data(self):
        """保存测试数据"""
        num_test = len(self.test_im_idx)
        fmri_array = np.zeros((num_test, self.num_voxel))
        stim_array = np.zeros((num_test, 425, 425, 3))
        
        for i, idx in enumerate(self.test_im_idx):
            stim_array[i] = self.stim[idx]
            fmri_array[i] = self.fmri[sorted(self.sig_test[idx])].mean(0)
            
        np.save(f'{self.savedir}/nsd_test_fmriavg_nsdgeneral_{self.subject}.npy', fmri_array)
        np.save(f'{self.savedir}/nsd_test_stim_{self.subject}.npy', stim_array)
        print("Test data are saved.")


    def save_caption_data(self):
        """保存字幕数据"""
        # 保存训练集字幕
        train_captions = self._process_captions(self.train_im_idx)
        np.save(f'{self.savedir}/nsd_train_cap_{self.subject}.npy', train_captions)
        print("Training captions are saved.")
        
        # 保存测试集字幕
        test_captions = self._process_captions(self.test_im_idx)
        np.save(f'{self.savedir}/nsd_test_cap_{self.subject}.npy', test_captions)
        print("Test captions are saved.")
    def _process_captions(self, im_idx):
        """处理字幕数据的辅助函数"""
        captions = np.empty((len(im_idx), 5), dtype=object)
        for i, nsdId in enumerate(tqdm.tqdm(im_idx)):
            cocoId = self.stim_info[self.stim_info.nsdId==nsdId].cocoId.values[0]
            split = self.stim_info[self.stim_info.nsdId==nsdId].cocoSplit.values[0]
            
            if split == "train2017":
                cap = self.caption_train_df[self.caption_train_df.image_id==cocoId].caption.values
            else:
                cap = self.caption_val_df[self.caption_val_df.image_id==cocoId].caption.values
            captions[i,:] = cap[:5]
        return captions

    def loadmat(self,filename):
        """加载.mat文件的辅助函数"""
        '''
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        '''
        '''
        重写loadmat函数，修复部分问题，将复杂的.mat文件数据转换为字典dict或列表list
        '''
        def _check_keys(d):
            '''
            checks if entries in dictionary are mat-objects. If yes
            todict is called to change them to nested dictionaries
            '''
            '''
            递归检查加载数据的键值对
            '''
            for key in d:
                if isinstance(d[key], spio.matlab.mat_struct):
                    d[key] = _todict(d[key])  # 若为mat_struct则调用_todicit
            return d

        def _todict(matobj):
            '''
            A recursive function which constructs from matobjects nested dictionaries
            '''
            '''
            递归检查，将mat_struct转换为dict
            '''
            d = {}
            for strg in matobj._fieldnames:
                elem = matobj.__dict__[strg]
                if isinstance(elem, spio.matlab.mat_struct):
                    d[strg] = _todict(elem)
                elif isinstance(elem, np.ndarray):
                    d[strg] = _tolist(elem)
                else:
                    d[strg] = elem
            return d

        def _tolist(ndarray):
            '''
            A recursive function which constructs lists from cellarrays
            (which are loaded as numpy ndarrays), recursing into the elements
            if they contain matobjects.
            '''
            '''
            递归检查，将mat_struct转换为list
            '''
            elem_list = []
            for subj_elem in ndarray:
                if isinstance(subj_elem, spio.matlab.mat_struct):
                    elem_list.append(_todict(subj_elem))
                elif isinstance(subj_elem, np.ndarray):
                    elem_list.append(_tolist(subj_elem))
                else:
                    elem_list.append(subj_elem)
            return elem_list
        # 加载.mat文件数据
        data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return _check_keys(data) #调用_check_keys


def main():
    """主函数"""
    processor = NSDDataProcessor()
    
    print("开始数据处理...")
    processor.load_stimulus_info()
    processor.prepare_trial_indices()
    
    print("加载fMRI数据...")
    processor.load_fmri_data()
    
    print("加载视觉刺激数据...")
    processor.load_stimulus_data()
    
    print("保存训练数据...")
    processor.save_training_data()
    
    print("保存测试数据...")
    processor.save_test_data()
    
    print("处理并保存字幕数据...")
    processor.save_caption_data()
    
    print("数据处理完成！")

if __name__ == "__main__":
    main() 