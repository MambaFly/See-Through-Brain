import torch
import numpy as np
from PIL import Image
import torchvision
from transformers import CLIPProcessor, CLIPModel
import os
import pandas as pd
import pickle

def load_data():
    # 加载图像和字幕数据，添加 allow_pickle=True 参数
    images = np.load('./processed_data/subj01/nsd_test_stim_subj01.npy', allow_pickle=True)
    captions = np.load('./processed_data/subj01/nsd_test_cap_subj01.npy', allow_pickle=True)
    return images, captions

def select_best_captions():
    # 加载CLIP模型和处理器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    to_pil = torchvision.transforms.ToPILImage()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # 加载数据
    images, captions = load_data()
    
    # 存储最佳字幕的列表
    best_captions = []
    
    print(len(images))
    print(len(captions))  
    # 对每张图像进行处理
    for i in range(len(images)):
        image = to_pil(images[i])
        if i == 0:
            print( "type(images[i]):", type(images[i]))
            print( "type(image):", type(image))
        current_captions = captions[i][:5]  # 获取当前图像对应的5个字幕
        
        # 确保captions是字符串列表
        if isinstance(current_captions, np.ndarray):
            current_captions = current_captions.tolist()
        
        # 确保每个caption都是字符串类型
        current_captions = [str(cap) for cap in current_captions]
        
        # 打印第一组数据的调试信息
        if i == 0:
            print("第一组字幕类型:", type(current_captions))
            print("第一个字幕类型:", type(current_captions[0]))
            print("字幕示例:", current_captions[0])
        
        # 处理图像和文本
        inputs = processor(
            images=image,
            text=current_captions,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # 计算相似度
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            similarities = logits_per_image[0].cpu().numpy()
        
        # 选择相似度最高的字幕
        best_caption_idx = np.argmax(similarities)
        best_caption = current_captions[best_caption_idx]
        best_captions.append(best_caption)
        
        if (i + 1) % 100 == 0:
            print(f"已处理 {i+1} 张图像")
    
    # with open(f"./decoded_captions/subj01/test_captions.sav", "wb") as f:
    #     pickle.dump(best_captions, f)

    # 修改保存路径并添加CSV保存
    output_npy_path = './processed_data/subj01/best_captions.npy'
    output_csv_path = './processed_data/subj01/best_captions.csv'
    
    # 保存NPY文件
    np.save(output_npy_path, np.array(best_captions))
    print(f"最佳字幕已保存至: {output_npy_path}")
    
    # 保存CSV文件
    df = pd.DataFrame({'caption': best_captions})
    df.to_csv(output_csv_path, index=False)
    print(f"最佳字幕CSV文件已保存至: {output_csv_path}")

if __name__ == "__main__":
    # select_best_captions()
    caps = np.load('./processed_data/subj01/best_captions.npy', allow_pickle=True)
    print(type(caps.tolist()))
    print(len(caps.tolist()))
    with open(f"decoded_captions/subj01/test_captions.sav", "rb") as f:
        test_captions = pickle.load(f)
    print(type(test_captions))
    print(len(test_captions))