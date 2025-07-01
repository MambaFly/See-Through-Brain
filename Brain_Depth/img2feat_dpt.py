import argparse, os
from tqdm import tqdm
import torch
import numpy as np
import PIL
from transformers import AutoImageProcessor, DPTForDepthEstimation
import sys
from nsd_access.nsda import NSDAccess
from PIL import Image
import cv2

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--imgidx",
        required=True,
        nargs="*",
        type=int,
        help="start and end imgs"
    )

    opt = parser.parse_args()
    imgidx = opt.imgidx
    gpu = 0
    resolution = 512
    nsda = NSDAccess('../Brain-Diffusion/data/nsd')

    # Save Directories
    os.makedirs(f'../data/nsdfeat/dpt/', exist_ok=True)
    for i in range(4):
        os.makedirs(f'../data/nsdfeat/dpt_emb{i}/', exist_ok=True)

    image_processor = AutoImageProcessor.from_pretrained("./dpt_large")
    model = DPTForDepthEstimation.from_pretrained("./dpt_large")
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    

    for s in tqdm(range(imgidx[0],imgidx[1])):
        print(f"Now processing image {s:06}")
        img_arr = nsda.read_images(s)
        gray_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        image = Image.fromarray(gray_img).convert("RGB").resize((resolution, resolution), resample=PIL.Image.LANCZOS)
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs,output_hidden_states=True)
            predicted_depth = outputs.predicted_depth
        hidden_states = [
            feature.to('cpu').detach().numpy() for idx, feature in enumerate(outputs.hidden_states[1:]) if idx in model.config.backbone_out_indices
            ]

        predicted_depth = predicted_depth.to('cpu').detach().numpy()

        for idx, dpt_idx in enumerate(model.config.backbone_out_indices):
            np.save(f'../data/nsdfeat/dpt_emb{idx}/{s:06}.npy',hidden_states[idx])
        np.save(f'../data/nsdfeat/dpt/{s:06}.npy',predicted_depth)

if __name__ == "__main__":
    main()
