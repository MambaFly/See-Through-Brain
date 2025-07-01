'''
引入相关包
'''
from transformers import AutoProcessor
from modeling_git import GitForCausalLM, GitModel, GitForCausalLMClipEmb
import requests
from PIL import Image
import numpy as np
import os
import glob
from os.path import join as opj
import h5py  
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import json
from torchsummary import summary
import torchvision
import tqdm
from sklearn.linear_model import RidgeCV
import pickle

'''
相关文件的路径
'''
#根目录
base_path="./"
#时间序列数据和大脑响应数据
timeseries_path=opj(base_path,"nsddata_timeseries")
betas_path=opj(base_path,"nsddata_betas")
#刺激数据（图像）
stimuli_path=opj(base_path,"nsddata_stimuli","stimuli","nsd")
stim_file_path=opj(stimuli_path,"nsd_stimuli.hdf5")
#特定受试者（subj01）的数据
subj="subj01"

mod="func1pt8mm"
subj_data_path=opj(timeseries_path,"ppdata",subj,mod,"timeseries")
subj_betas_path=opj(betas_path,"ppdata",subj,mod,"betas_assumehrf")
#经过感兴趣区域（ROI）提取处理的大脑响应数据
subj_betas_roi_extracted_path=opj(base_path,"processed_roi",subj,mod)
#指向实验设计、刺激信息和字幕文件的路径
stim_order_path=opj(base_path,"nsddata","experiments","nsd","nsd_expdesign.mat")
stim_info_path=opj(base_path,"nsddata","experiments","nsd","nsd_stim_info_merged.csv")
stim_captions_train_path=opj(base_path,"nsddata_stimuli","stimuli","nsd","annotations",f"captions_train2017.json")
stim_captions_val_path=opj(base_path,"nsddata_stimuli","stimuli","nsd","annotations",f"captions_val2017.json")
#指向处理后的数据集
processed_data=opj(base_path,"processed_data",subj)

#训练集的fMRI数据、图像数据和字幕数据
fmri_train_data=opj(processed_data,f"nsd_train_fmriavg_nsdgeneral_{subj}.npy")
imgs_train_data=opj(processed_data,f"nsd_train_stim_{subj}.npy")
captions_train_data=opj(processed_data, f"nsd_train_cap_{subj}.npy")
#测试集的相应数据
fmri_test_data=opj(processed_data,f"nsd_test_fmriavg_nsdgeneral_{subj}.npy")
imgs_test_data=opj(processed_data,f"nsd_test_stim_{subj}.npy")
captions_test_data=opj(processed_data, f"nsd_test_cap_{subj}.npy")

'''
控制参数
'''
compute_dataset=True
train=True
adjust=True

'''
加载和处理NSD相关数据的函数
输入：fMRI，img，caption
输出：fMRI，img，caption
'''
class NSDDataset(Dataset):
    
    def __init__(self, fmri_data,imgs_data,caption_data,transforms=None):
        self.fmri_data=np.load(fmri_data)
        self.imgs_data=np.load(imgs_data).astype(np.uint8)
        self.caption_data=np.load(caption_data,allow_pickle=True)
        self.transforms=transforms
        
    def __len__(self):
        return  len(self.fmri_data)
    
    def __getitem__(self,idx):
        fmri=torch.tensor(self.fmri_data[idx])
        img=Image.fromarray(self.imgs_data[idx])
        
        if self.transforms:
            img=self.transforms(img)
        
        caption=self.caption_data[idx][0] #cambiare se ne voglio altre
        
        return fmri,img,caption

'''
创建数据集NSDDataset和数据加载器DataLoader
'''
# 作为参数，将图像数据转换为PyTorch张量
tr=torchvision.transforms.ToTensor() 
#分别创建了训练集和测试集的NSDDataset实例
train_dataset=NSDDataset(fmri_train_data,imgs_train_data,captions_train_data,transforms=tr)
test_dataset=NSDDataset(fmri_test_data,imgs_test_data,captions_test_data,transforms=tr)
#批次大小128
BS=128
#创建了两个DataLoader实例，用于在训练和测试过程中迭代数据
train_dataloader=DataLoader(train_dataset,BS,shuffle=True)
test_dataloader=DataLoader(test_dataset,BS,shuffle=False)

'''
设置预训练模型
'''
# 加载预训练模型

#用于处理输入图像数据
processor = AutoProcessor.from_pretrained("./git-base-coco")  
#结合了图像编码器和文本解码器，用于生成文本
model = GitForCausalLMClipEmb.from_pretrained("./git-base-coco")
#用于生成文本，但不涉及图像处理（没有图像编码器）
model_base = GitForCausalLM.from_pretrained("./git-base-coco")

# 指定模型运行在GPU
device="cuda:0"
model.to(device)
model_base.to(device)

to_pil=torchvision.transforms.ToPILImage()


# 从model中提取图像编码器部分
vision_encoder=model.git.image_encoder

'''
为模型训练准备数据，
将图像数据转换为模型可以理解的嵌入表示，
并与fMRI数据和字幕数据一起存储
'''
if compute_dataset:
    #存储训练数据
    train_fmri=[]
    train_imgs=[]
    train_captions=[]
    train_clip_img_embeds=[]

    # 对于数据加载器中的每个批次（包含fMRI数据x、图像数据y和字幕数据c）
    for x,y,c in tqdm.tqdm(train_dataloader):

        train_fmri.append(x)
        
        train_imgs.append(y)
        
        train_captions+=list(c)
        
        #encode images in autoencoder and save z representation
        # 将图像编码并保存表征z
        with torch.no_grad():
            
            #encode images in CLIP
            #处理图像数据y中的每个图像
            image = [to_pil(y[i]) for i in range(y.size(0))]
            pixel_values= processor(images=image, return_tensors="pt").pixel_values.to(device)
            #使用图像编码器，获得图像的嵌入表征
            image_features=vision_encoder(pixel_values).last_hidden_state.cpu()
            train_clip_img_embeds.append(image_features)
            
    print(len(train_clip_img_embeds))
    print(train_clip_img_embeds[0].shape)
    train_clip_img_embeds = torch.cat(train_clip_img_embeds,axis=0)
    print(train_clip_img_embeds.shape)
    train_fmri = torch.cat(train_fmri,axis=0)
    train_imgs = torch.cat(train_imgs,axis=0)

'''
为模型的测试阶段准备数据，
确保测试数据以与训练数据相同的方式被处理。
模型可以在测试阶段使用这些数据来评估其性能
'''
if compute_dataset:
    test_fmri=[]
    test_imgs=[]
    test_captions=[]
    test_clip_img_embeds=[]


    for x,y,c in tqdm.tqdm(test_dataloader):

        #save fMRI data
        test_fmri.append(x)

        #save img data
        test_imgs.append(y)

        #save caption data
        test_captions+=list(c)

        #encode images in autoencoder and save z representation
        with torch.no_grad():
            
            #encode images in CLIP
            image = [to_pil(y[i]) for i in range(y.size(0))]
            pixel_values= processor(images=image, return_tensors="pt").pixel_values.to(device)
            image_features=vision_encoder(pixel_values).last_hidden_state.cpu()
            test_clip_img_embeds.append(image_features)

           
    test_clip_img_embeds = torch.cat(test_clip_img_embeds,axis=0)
    test_fmri = torch.cat(test_fmri,axis=0)
    test_imgs = torch.cat(test_imgs,axis=0)


'''
数据集的保存和加载
确保无论是从头开始计算数据集还是从之前保存的数据加载，
都能够正确地准备训练和测试数据
这样，模型就可以在训练和测试阶段使用这些数据
'''
# 数据集的保存
if compute_dataset:
    os.makedirs(f"models/{subj}",exist_ok=True)
    
    ## train
    #将训练集的fMRI数据、图像嵌入表示和图像数据保存为PyTorch张量文件（.pt）
    torch.save(train_fmri,f"models/{subj}/train_fmri.pt")
    torch.save(train_clip_img_embeds,f"models/{subj}/train_clip_img_embeds.pt")
    torch.save(train_imgs,f"models/{subj}/train_imgs.pt")
    #将训练集字幕数据保存为二进制文件（.sav）
    with open(f"models/{subj}/train_captions.sav","wb") as f:
        pickle.dump(train_captions,f)
        
    print("saved training stuff")
    
    ## test  （同train）
    torch.save(test_fmri,f"models/{subj}/test_fmri.pt")
    torch.save(test_clip_img_embeds,f"models/{subj}/test_clip_img_embeds.pt")
    torch.save(test_imgs,f"models/{subj}/test_imgs.pt")
        
    with open(f"models/{subj}/test_captions.sav","wb") as f:
        pickle.dump(test_captions,f)
    
    print("saved testing stuff")
    

# 数据集的加载
else:
    if subj=="subj01_good2":
        subj="subj01"
    ## train
    train_fmri=torch.load(f"models/{subj}/train_fmri.pt")
    train_clip_img_embeds= torch.load(f"models/{subj}/train_clip_img_embeds.pt")
    train_imgs=torch.load(f"models/{subj}/train_imgs.pt")
        
    with open(f"models/{subj}/train_captions.sav","rb") as f:
        train_captions=pickle.load(f)

    ## test
    test_fmri=torch.load(f"models/{subj}/test_fmri.pt")
    test_clip_img_embeds= torch.load(f"models/{subj}/test_clip_img_embeds.pt")
    test_imgs=torch.load(f"models/{subj}/test_imgs.pt")
    
    with open(f"models/{subj}/test_captions.sav","rb") as f:
        test_captions=pickle.load(f)

'''
数据标准化
确保模型在训练过程中不会因为输入数据的尺度差异而产生偏差，
有助于提高模型的泛化能力
'''
# 计算均值和标准差
train_fmri_mean=torch.mean(train_fmri,axis=0)
train_fmri_std=torch.std(train_fmri,axis=0)
# 对训练集和测试集标准化
train_fmri_norm=(train_fmri-train_fmri_mean)/train_fmri_std
test_fmri_norm=(test_fmri-train_fmri_mean)/train_fmri_std

'''
训练一个模型，
该模型能够将大脑活动映射到图像嵌入，
用于后续的字幕生成
'''
# 限制嵌入特征的大小
max_len_img=197

if train:
    #存储训练好的大脑到图像嵌入的映射模型
    brain_to_img_emb=[]
    alphas = [1e3,3e3,9e3,1e4,3e4,9e4,1e5,3e5,9e5,1e6]
    #使用Ridge回归模型（岭回归）来学习大脑活动（train_fmri_norm���
    #到图像嵌入（train_clip_img_embeds[:,i,:]）之间的映射
    for i in tqdm.tqdm(range(max_len_img)):
        m=RidgeCV(alphas)  # alpha=6e4是正则化参数
        m.fit(train_fmri_norm.numpy(),train_clip_img_embeds[:,i,:].numpy())
        brain_to_img_emb.append(m)

# 保存或加载 大脑-图像 映射模型
if train:
    os.makedirs(f"models/{subj}/decoding",exist_ok=True)
    for i in range(max_len_img):
        filename = f'brain_to_img_emb_ridge_{i}.sav'
        with open(opj(f"models/{subj}/decoding",filename), 'wb') as f:
            pickle.dump(brain_to_img_emb[i], f) 
else:
    brain_to_img_emb=[]
    for i in range(max_len_img):
        filename = f'brain_to_img_emb_ridge_{i}.sav'
        with open(opj(f"models/{subj}/decoding",filename), 'rb') as f:
            p=pickle.load(f)
            brain_to_img_emb.append(p)

#使用训练好的映射模型预测图像嵌入
if train:
    img_emb_train=[]
    for i in tqdm.tqdm(range(max_len_img)):
        emb=torch.tensor(brain_to_img_emb[i].predict(train_fmri_norm.numpy()))
        img_emb_train.append(emb)
    img_emb_train=torch.stack(img_emb_train,1)
    
# 将训练的图像嵌入 和预测的图像嵌入 标准化
# 保存或加载 嵌入数据
if train:
    
    train_clip_img_embeds_mean=train_clip_img_embeds.mean(0)
    train_clip_img_embeds_std=train_clip_img_embeds.std(0)
    
    pred_clip_img_embeds_mean=img_emb_train.mean(0)
    pred_clip_img_embeds_std=img_emb_train.std(0)
    
    torch.save(train_clip_img_embeds_mean, opj(f"models/{subj}","train_clip_img_embeds_mean.pt"))
    torch.save(train_clip_img_embeds_std, opj(f"models/{subj}","train_clip_img_embeds_std.pt"))
    torch.save(pred_clip_img_embeds_mean, opj(f"models/{subj}","pred_clip_img_embeds_mean.pt"))
    torch.save(pred_clip_img_embeds_std, opj(f"models/{subj}","pred_clip_img_embeds_std.pt"))
    
else:
    train_clip_img_embeds_mean=torch.load(opj(f"models/{subj}","train_clip_img_embeds_mean.pt"))
    train_clip_img_embeds_std=torch.load(opj(f"models/{subj}","train_clip_img_embeds_std.pt"))
   
    pred_clip_img_embeds_mean=torch.load(opj(f"models/{subj}","pred_clip_img_embeds_mean.pt"))
    pred_clip_img_embeds_std=torch.load(opj(f"models/{subj}","pred_clip_img_embeds_std.pt"))

'''
为测试集生成图像嵌入，
并对这些嵌入进行调整，
以便它们与训练集的分布相匹配
'''

# 存储测试集的图像嵌入
img_emb_test=[]
#使用大脑-图像嵌入的映射模型来预测测试集fMRI数据的图像嵌入
for i in tqdm.tqdm(range(max_len_img)):
    emb=torch.tensor(brain_to_img_emb[i].predict(test_fmri_norm.numpy()))
    # 将预测的图像嵌入添加到img_emb_test列表
    img_emb_test.append(emb)
    
#将列表中的图像嵌入堆叠成一个二维张量
img_emb_test=torch.stack(img_emb_test,1)

#调整测试集的图像嵌入 以匹配训练集的分布
if adjust:
    img_emb_test_adj=(img_emb_test-img_emb_test.mean(0))/(img_emb_test.std(0))
    print(img_emb_test_adj.shape)
    print(train_clip_img_embeds_std.shape)
    img_emb_test_adj=train_clip_img_embeds_std*img_emb_test_adj+train_clip_img_embeds_mean

'''
生成字幕
一组直接从图像生成，另一组从大脑活动映射到的图像嵌入生成。
这可以帮助评估模型在理解视觉信息和生成字幕方面的能力
'''

# 存储从图像和大脑活动生成的字幕
captions_from_images=[]
captions_from_brain=[]

with torch.no_grad():
    # 对于测试集中的每个批次（大小为BS）-> 改成逐个图像生成
    for i in tqdm.tqdm(range(len(test_imgs))):
        
        #get reference images
        #获取当前批次的所有图像
        test_img = test_imgs[i]
        # 使用to_pil将图像数据转换为PIL图像
        img=to_pil(test_img)

        #compute captions from images
        #使用processor将图像转换为pixel_values
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        #使用model_base从图像生成字幕，generate方法生成字幕的ID
        generated_ids = model_base.generate(pixel_values=pixel_values.to(device), max_length=25)
        #batch_decode方法将这些ID转换为可读的文本
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
        #将生成的字幕添加到captions_from_images列表
        captions_from_images+=generated_caption

        #compute captions from brain
        # 对于同一批次的图像，使用model（结合了图像编码器和文本解码器的模型）
        # 从调整后的图像嵌入img_emb_test_adj生成字幕
        pixel_values=img_emb_test_adj[i].unsqueeze(0) #增加维度，从而能作为模型输入
        generated_ids = model.generate(pixel_values=pixel_values.to(device).float(), max_length=25)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
        captions_from_brain+=generated_caption

'''
字幕保存
'''
os.makedirs(f"decoded_captions/{subj}",exist_ok=True)

with open(f"decoded_captions/{subj}/test_captions.sav","wb") as f:
    pickle.dump(test_captions,f)
    
with open(f"decoded_captions/{subj}/image_captions.sav","wb") as f:
    pickle.dump(captions_from_images,f) 

with open(f"decoded_captions/{subj}/brain_captions.sav","wb") as f:
    pickle.dump(captions_from_brain,f)

'''解码大脑字幕保存'''
data = captions_from_brain
df = pd.DataFrame(data)
savedir = f"./decoded_captions/{subj}"
os.makedirs(savedir, exist_ok=True)
df.to_csv(f'{savedir}/captions_brain.csv', sep='\t', header=False, index=False)
