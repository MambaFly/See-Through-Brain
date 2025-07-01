import argparse, os
import numpy as np
import scipy.io
from tqdm import tqdm
import torch
import PIL
from transformers import AutoImageProcessor, DPTForDepthEstimation
from nsd_access.nsda import NSDAccess
from PIL import Image
import cv2
import h5py

def main():

    # 固定参数
    subject = 'subj01'
    imgids = [0, 1000]
    gpu = 0
    resolution = 512
    imsize = (512, 512)

    # 设置路径
    nsda = NSDAccess('../Brain-Diffusion/data/nsd')
    
    # 加载nsd实验设计数据获取测试索引
    nsd_expdesign = scipy.io.loadmat('../Brain-Diffusion/data/nsd/nsddata/experiments/nsd/nsd_expdesign.mat')
    # 注意：这是1-base索引，需要减1
    sharedix = nsd_expdesign['sharedix'] - 1
    print(f'测试集大小：{len(sharedix[0])}')
    
    # 加载刺激文件，类似于diffusion_decoding_v1.py
    sf = h5py.File(nsda.stimuli_file, 'r')
    sdataset = sf.get('imgBrick')
    
    # 加载受试者特定的刺激平均值
    stims_ave = np.load(f'../Brain-Diffusion/data/stim/{subject}/{subject}_stims_ave.npy')
    
    # 区分训练集和测试集
    tr_idx = np.zeros_like(stims_ave)
    for idx, s in enumerate(stims_ave):
        if s in sharedix:
            tr_idx[idx] = 0
        else:
            tr_idx[idx] = 1
    
    # 创建保存目录，按受试者分类
    outdir = f'../data/nsdfeat/depth_testpic/'
    os.makedirs(outdir, exist_ok=True)
    
    org_dir = f'{outdir}/org/'
    os.makedirs(org_dir, exist_ok=True)
    
    vis_dir = f'{outdir}/vis/'
    os.makedirs(vis_dir, exist_ok=True)
    
    emb_dir = f'{outdir}/emb/'
    os.makedirs(emb_dir, exist_ok=True)
    
    # 加载DPT模型
    image_processor = AutoImageProcessor.from_pretrained("./dpt_large")
    model = DPTForDepthEstimation.from_pretrained("./dpt_large")
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    # 处理指定范围内的测试图像
    print(f"开始处理测试图像（从索引 {imgids[0]} 到 {imgids[1]-1}）...")
    
    for imgidx in tqdm(range(imgids[0], imgids[1])):
        # print(f"\n ...正在处理图像 {imgidx}... \n")
        
        # 获取测试图像索引，类似于diffusion_decoding_v1.py中的方法
        imgidx_te = np.where(tr_idx==0)[0][imgidx]  # 提取测试图像索引
        idx73k = stims_ave[imgidx_te]
        
        # 保存原始图像供参考
        Image.fromarray(np.squeeze(sdataset[idx73k,:,:,:]).astype(np.uint8)).save(
            os.path.join(org_dir, f"{imgidx:05}_org.png"))
        
        # 读取图像
        img_arr = nsda.read_images(idx73k)
        gray_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        image = Image.fromarray(gray_img).convert("RGB").resize((resolution, resolution), resample=PIL.Image.LANCZOS)
        
        # 进行深度预测
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            predicted_depth = outputs.predicted_depth
        
        # 保存深度数据
        depth_np = predicted_depth.to('cpu').detach().numpy()
        np.save(f'{emb_dir}/{imgidx:05}.npy', depth_np)
        
        # 保存可视化深度图
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=imsize,
            mode="bicubic",
            align_corners=False,
        )
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth_img = Image.fromarray(formatted)
        depth_img.save(f'{vis_dir}/{imgidx:05}.png')
    
    print(f"所有测试图像的深度预测已完成，结果保存在 {outdir}")

if __name__ == "__main__":
    main() 