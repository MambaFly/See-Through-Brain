import sys
# 清除任何可能导致冲突的路径
for i in range(len(sys.path)-1, -1, -1):
    if "StableDiffusionReconstruction" in sys.path[i] or "diffusion_sd1" in sys.path[i]:
        sys.path.pop(i)
# 添加正确的路径
sys.path.insert(0, '/data/zlhu/NeuroAI/Eye-of-Brain/Brain-Diffusion/decoding_sd2/stablediffusion')

import h5py
import scipy.io
from nsd_access.nsda import NSDAccess
import argparse, os
from tqdm import tqdm, trange
from torch import autocast
from contextlib import nullcontext
import torch
import numpy as np
import pandas as pd
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# from ldm.data.util import AddMiDaS
parser = argparse.ArgumentParser()
parser.add_argument(
        "--subject",
        required=True,
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )


def initialize_model(config, ckpt,device):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)
    model = model.to(device)
    sampler = DDIMSampler(model)
    return sampler


# Set parameters
opt = parser.parse_args()
subject=opt.subject
seed_everything(42)
imgidxs = [0, 10]
gpu = 0
torch.cuda.set_device(gpu)
config = './stablediffusion/configs/stable-diffusion/v2-midas-inference.yaml'
ckpt = './stablediffusion/models/512-depth-ema.ckpt'
steps = 50
scale = 5.0
eta = 0.0
strength = 0.8
num_samples= 1
callback=None
n_iter = 5

# Save Directories
outdir = f'../../Brain-Decoded/{subject}/image-text-depth'
os.makedirs(outdir, exist_ok=True)


precision = 'autocast'
precision_scope = autocast if precision == "autocast" else nullcontext

device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
sampler = initialize_model(config, ckpt,device)
model = sampler.model

'''采样策略'''
sampler.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=True)

assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'

t_enc = min(int(strength * steps), steps-1)
print(f"target t_enc is {t_enc} steps")


# Load Prediction (C, InitLatent, Depth(cc))
captdir = f'../../Brain-Caption/decoded_captions/{subject}'
dptdir = f'../../data/decoded/{subject}/dpt_fromemb/'
# gandir = f'../../decoded/gan_recon_img/all_layers/{subject}/streams/'

# C
captions = pd.read_csv(f'{captdir}/captions_brain.csv', sep='\t',header=None)


import h5py
import scipy.io
from nsd_access.nsda import NSDAccess
# Load NSD information

nsd_expdesign = scipy.io.loadmat('../../data/nsd/nsddata/experiments/nsd/nsd_expdesign.mat')

nsda = NSDAccess('../../data/nsd/')
sf = h5py.File(nsda.stimuli_file, 'r')
sdataset = sf.get('imgBrick')

stims_ave = np.load(f'../../data/stim/{subject}/{subject}_stims_ave.npy')

# Note that mos of them are 1-base index!
# This is why I subtract 1
sharedix = nsd_expdesign['sharedix'] -1 
tr_idx = np.zeros_like(stims_ave)
for idx, s in enumerate(stims_ave):
    if s in sharedix:
        tr_idx[idx] = 0
    else:
        tr_idx[idx] = 1

# Load NSD information
nsd_expdesign = scipy.io.loadmat('../../data/nsd/nsddata/experiments/nsd/nsd_expdesign.mat')

nsda = NSDAccess('../../data/nsd/')
sf = h5py.File(nsda.stimuli_file, 'r')
sdataset = sf.get('imgBrick')

stims_ave = np.load(f'../../data/stim/{subject}/{subject}_stims_ave.npy')

# Note that mos of them are 1-base index!
# This is why I subtract 1
sharedix = nsd_expdesign['sharedix'] -1 
tr_idx = np.zeros_like(stims_ave)
for idx, s in enumerate(stims_ave):
    if s in sharedix:
        tr_idx[idx] = 0
    else:
        tr_idx[idx] = 1

### decoding NO.2 第二版

def load_img_from_arr(img_arr):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = 512, 512
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
for imgidx in range(1000):

    print(f"now is img{imgidx}.../n")
    # Load z (Image)
    imgidx_te = np.where(tr_idx==0)[0][imgidx] # Extract test image index
    idx73k= stims_ave[imgidx_te]
    Image.fromarray(np.squeeze(sdataset[idx73k,:,:,:]).astype(np.uint8)).save(
        os.path.join(outdir, f"{imgidx:05}_org.png"))
    
    c = model.get_learned_conditioning(captions.iloc[imgidx][0]).to('cuda')
    cc = torch.Tensor(np.load(f'{dptdir}/{imgidx:06}.npy')).to('cuda')
    
    # # Generate image from Text + GAN + Depth
    # shenpath = f'{gandir}/recon_image_normalized-VGG19-fc8-{subject}-streams-{imgidx:06}.tiff'
    # init_image = Image.open(shenpath).resize((512,512))
    
    # Generate image from Text + Depth
    
    roi_latent = 'early'
    scores_latent = np.load(f'../data/decoded/{subject}/{subject}_{roi_latent}_brain_embs_init_latent.npy')
    imgarr = torch.Tensor(scores_latent[imgidx,:].reshape(4,40,40)).unsqueeze(0).to('cuda')
    
    # Generate image from Z
    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                x_samples = model.decode_first_stage(imgarr)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    
                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
    
    im = Image.fromarray(x_sample.astype(np.uint8)).resize((512,512))
    im = np.array(im)
    init_image = load_img_from_arr(im).to('cuda')
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
    
    base_count = 0
    with torch.no_grad():
        for n in trange(n_iter, desc="Sampling"):
            torch.autocast("cuda")
            
            c_cat = list()
            c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)
    
            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
    
            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
    
            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(
                init_latent, torch.tensor([t_enc] * num_samples).to(model.device))
    
            # decode it
            samples = sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale,
                                    unconditional_conditioning=uc_full, callback=callback)
            x_samples_ddim = model.decode_first_stage(samples)
            result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
            Image.fromarray(result[0,:,:,:].astype(np.uint8)).save(
                os.path.join(outdir, f"{imgidx:05}_{base_count:03}.png"))   
            base_count += 1