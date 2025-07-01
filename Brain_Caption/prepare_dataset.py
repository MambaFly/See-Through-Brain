'''
数据预处理模块
功能：加载原始数据，创建数据集，并保存处理后的数据供后续使用
'''

from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from modeling_git import GitForCausalLM, GitModel, GitForCausalLMClipEmb
from PIL import Image
import numpy as np
import os
from os.path import join as opj
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import tqdm
import pickle

# 数据集类定义
class NSDDataset(Dataset):
    def __init__(self, fmri_data, imgs_data, caption_data, transforms=None):
        self.fmri_data = np.load(fmri_data)
        self.imgs_data = np.load(imgs_data).astype(np.uint8)
        self.caption_data = np.load(caption_data, allow_pickle=True)
        self.transforms = transforms
        
    def __len__(self):
        return len(self.fmri_data)
    
    def __getitem__(self, idx):
        fmri = torch.tensor(self.fmri_data[idx])
        img = Image.fromarray(self.imgs_data[idx])
        
        if self.transforms:
            img = self.transforms(img)
        
        caption = self.caption_data[idx][0]
        return fmri, img, caption

def main():
    # 设置路径
    base_path = "./"
    subj = "subj02"
    processed_data = opj(base_path, "processed_data", subj)

    # 数据文件路径
    fmri_train_data = opj(processed_data, f"nsd_train_fmriavg_nsdgeneral_{subj}.npy")
    imgs_train_data = opj(processed_data, f"nsd_train_stim_{subj}.npy")
    captions_train_data = opj(processed_data, f"nsd_train_cap_{subj}.npy")
    fmri_test_data = opj(processed_data, f"nsd_test_fmriavg_nsdgeneral_{subj}.npy")
    imgs_test_data = opj(processed_data, f"nsd_test_stim_{subj}.npy")
    captions_test_data = opj(processed_data, f"nsd_test_cap_{subj}.npy")

    # 创建数据集和数据加载器
    tr = torchvision.transforms.ToTensor()
    train_dataset = NSDDataset(fmri_train_data, imgs_train_data, captions_train_data, transforms=tr)
    test_dataset = NSDDataset(fmri_test_data, imgs_test_data, captions_test_data, transforms=tr)
    
    print(f"训练集数据量: {len(train_dataset)}条")
    print(f"测试集数据量: {len(test_dataset)}条")
    
    BS = 128
    train_dataloader = DataLoader(train_dataset, BS, shuffle=True)
    test_dataloader = DataLoader(test_dataset, BS, shuffle=False)

    # 加载预训练模型
    device = "cuda:0"
    processor = AutoProcessor.from_pretrained("./git-base-coco")
    model = AutoModelForCausalLM.from_pretrained("./git-base-coco")
    # model = GitForCausalLMClipEmb.from_pretrained("./git-large-coco")
    model.to(device)
    vision_encoder = model.git.image_encoder
    to_pil = torchvision.transforms.ToPILImage()

    # 处理训练数据
    train_fmri = []
    train_imgs = []
    train_captions = []
    train_clip_img_embeds = []

    print("Processing training data...")
    for x, y, c in tqdm.tqdm(train_dataloader):
        train_fmri.append(x)
        train_imgs.append(y)
        train_captions += list(c)
        
        with torch.no_grad():
            image = [to_pil(y[i]) for i in range(y.size(0))]
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            image_features = vision_encoder(pixel_values).last_hidden_state.cpu()
            train_clip_img_embeds.append(image_features)

    train_clip_img_embeds = torch.cat(train_clip_img_embeds, axis=0)
    train_fmri = torch.cat(train_fmri, axis=0)
    train_imgs = torch.cat(train_imgs, axis=0)

    # 处理测试数据
    test_fmri = []
    test_imgs = []
    test_captions = []
    test_clip_img_embeds = []

    print("Processing test data...")
    for x, y, c in tqdm.tqdm(test_dataloader):
        test_fmri.append(x)
        test_imgs.append(y)
        test_captions += list(c)
        
        with torch.no_grad():
            image = [to_pil(y[i]) for i in range(y.size(0))]
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            image_features = vision_encoder(pixel_values).last_hidden_state.cpu()
            test_clip_img_embeds.append(image_features)

    test_clip_img_embeds = torch.cat(test_clip_img_embeds, axis=0)
    test_fmri = torch.cat(test_fmri, axis=0)
    test_imgs = torch.cat(test_imgs, axis=0)

    # 保存处理后的数据
    print("Saving processed data...")
    save_dir = f"models/{subj}"
    os.makedirs(save_dir, exist_ok=True)

    # 保存训练数据
    torch.save(train_fmri, f"{save_dir}/train_fmri.pt")
    torch.save(train_clip_img_embeds, f"{save_dir}/train_clip_img_embeds.pt")
    torch.save(train_imgs, f"{save_dir}/train_imgs.pt")
    with open(f"{save_dir}/train_captions.sav", "wb") as f:
        pickle.dump(train_captions, f)

    # 保存测试数据
    torch.save(test_fmri, f"{save_dir}/test_fmri.pt")
    torch.save(test_clip_img_embeds, f"{save_dir}/test_clip_img_embeds.pt")
    torch.save(test_imgs, f"{save_dir}/test_imgs.pt")
    with open(f"{save_dir}/test_captions.sav", "wb") as f:
        pickle.dump(test_captions, f)

    print("Data preprocessing completed!")

if __name__ == "__main__":
    main() 