'''
字幕生成模块
功能：加载训练好的模型，生成并保存字幕
'''

import os
from os.path import join as opj
import torch
import tqdm
import pickle
import pandas as pd
from transformers import AutoProcessor
from modeling_git import GitForCausalLM, GitForCausalLMClipEmb
import torchvision

def load_models_and_data(subj):
    """加载模型和数据"""
    # 加载预训练模型
    processor = AutoProcessor.from_pretrained("./git-base-coco")
    model = GitForCausalLMClipEmb.from_pretrained("./git-base-coco")
    model_base = GitForCausalLM.from_pretrained("./git-base-coco")
    
    # 加载测试数据
    test_fmri = torch.load(f"models/{subj}/test_fmri.pt")
    test_imgs = torch.load(f"models/{subj}/test_imgs.pt")
    
    with open(f"models/{subj}/test_captions.sav", "rb") as f:
        test_captions = pickle.load(f)
    
    # 加载统计量
    train_clip_img_embeds_mean = torch.load(opj(f"models/{subj}", "train_clip_img_embeds_mean.pt"))
    train_clip_img_embeds_std = torch.load(opj(f"models/{subj}", "train_clip_img_embeds_std.pt"))
    
    return processor, model, model_base, test_fmri, test_imgs, test_captions, train_clip_img_embeds_mean, train_clip_img_embeds_std

def load_brain_to_img_models(subj, max_len_img=197):
    """加载大脑到图像的映射模型"""
    brain_to_img_emb = []
    for i in range(max_len_img):
        filename = f'brain_to_img_emb_ridge_{i}.sav'
        with open(opj(f"models/{subj}/decoding", filename), 'rb') as f:
            p = pickle.load(f)
            brain_to_img_emb.append(p)
    return brain_to_img_emb

def generate_embeddings(test_fmri_norm, brain_to_img_emb, train_clip_img_embeds_mean, train_clip_img_embeds_std):
    """生成并调整图像嵌入"""
    max_len_img = 197
    img_emb_test = []
    
    print("Generating image embeddings...")
    for i in tqdm.tqdm(range(max_len_img)):
        emb = torch.tensor(brain_to_img_emb[i].predict(test_fmri_norm.numpy()))
        img_emb_test.append(emb)
    
    img_emb_test = torch.stack(img_emb_test, 1)
    
    # 调整嵌入
    img_emb_test_adj = (img_emb_test - img_emb_test.mean(0)) / (img_emb_test.std(0))
    print('---------------------')
    print(train_clip_img_embeds_std.shape)
    print(img_emb_test_adj.shape)
    print(train_clip_img_embeds_mean.shape  )
    print('---------------------')
    img_emb_test_adj = train_clip_img_embeds_std * img_emb_test_adj + train_clip_img_embeds_mean
    
    return img_emb_test_adj

def generate_captions(processor, model, model_base, test_imgs, img_emb_test_adj, device="cuda:0"):
    """生成字幕"""
    captions_from_images = []
    captions_from_brain = []
    to_pil = torchvision.transforms.ToPILImage()

    print("Generating captions...")
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(test_imgs))):
            # 从图像生成字幕
            test_img = test_imgs[i]
            img = to_pil(test_img)
            pixel_values = processor(images=img, return_tensors="pt").pixel_values
            generated_ids = model_base.generate(pixel_values=pixel_values.to(device), max_length=25)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
            captions_from_images += generated_caption

            # 从大脑活动生成字幕
            pixel_values = img_emb_test_adj[i].unsqueeze(0)
            generated_ids = model.generate(pixel_values=pixel_values.to(device).float(), max_length=25)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
            captions_from_brain += generated_caption
    
    return captions_from_images, captions_from_brain

def save_captions(subj, test_captions, captions_from_images, captions_from_brain):
    """保存生成的字幕"""
    save_dir = f"decoded_captions/{subj}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存所有字幕
    with open(f"{save_dir}/test_captions.sav", "wb") as f:
        pickle.dump(test_captions, f)
    with open(f"{save_dir}/image_captions.sav", "wb") as f:
        pickle.dump(captions_from_images, f)
    with open(f"{save_dir}/brain_captions.sav", "wb") as f:
        pickle.dump(captions_from_brain, f)
    
    # 保存大脑字幕为CSV
    df = pd.DataFrame(captions_from_brain)
    df.to_csv(f'{save_dir}/captions_brain.csv', sep='\t', header=False, index=False)

def main():
    subj = "subj02"
    device = "cuda:0"
    
    # 加载模型和数据
    print("Loading models and data...")
    processor, model, model_base, test_fmri, test_imgs, test_captions, \
    train_clip_img_embeds_mean, train_clip_img_embeds_std = load_models_and_data(subj)
    
    # 将模型移到GPU
    model.to(device)
    model_base.to(device)
    
    # 加载大脑到图像的映射模型
    brain_to_img_emb = load_brain_to_img_models(subj)
    
    # 标准化测试数据
    train_fmri_mean = torch.mean(test_fmri, axis=0)
    train_fmri_std = torch.std(test_fmri, axis=0)
    test_fmri_norm = (test_fmri - train_fmri_mean) / train_fmri_std
    
    # 生成图像嵌入
    img_emb_test_adj = generate_embeddings(test_fmri_norm, brain_to_img_emb, 
                                         train_clip_img_embeds_mean, train_clip_img_embeds_std)
    
    # 生成字幕
    captions_from_images, captions_from_brain = generate_captions(
        processor, model, model_base, test_imgs, img_emb_test_adj, device)
    
    # 保存结果
    print("Saving generated captions...")
    save_captions(subj, test_captions, captions_from_images, captions_from_brain)
    
    print("Caption generation completed!")

if __name__ == "__main__":
    main() 