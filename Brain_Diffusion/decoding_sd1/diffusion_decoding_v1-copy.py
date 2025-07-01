import h5py
from PIL import Image
import scipy.io
import argparse, os
import pandas as pd
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import trange
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import sys
from nsd_access.nsda import NSDAccess
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda(f"cuda:{gpu}")
    model.eval()
    return model

def load_img_from_arr(img_arr):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = 512, 512
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subject",
        required=True,
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )
    # parser.add_argument(
    #     "--method",
    #     required=True,
    #     type=str,
    #     help="init or text or gan",
    # )


    # Set parameters
    seed = 42
    seed_everything(seed)
    imgids = [0, 1000]
    gpu = 0
    method = 'text'

    # Set parameters
    opt = parser.parse_args()
    subject=opt.subject

    captdir = f'../../Brain-Caption/decoded_captions/{subject}'
    
    # Load NSD information
    nsd_expdesign = scipy.io.loadmat('../data/nsd/nsddata/experiments/nsd/nsd_expdesign.mat')
    
    # Note that mos of them are 1-base index!
    # This is why I subtract 1
    sharedix = nsd_expdesign['sharedix'] -1 
    
    nsda = NSDAccess('../data/nsd/')
    sf = h5py.File(nsda.stimuli_file, 'r')
    sdataset = sf.get('imgBrick')
    
    stims_ave = np.load(f'../data/stim/{subject}/{subject}_stims_ave.npy')

    tr_idx = np.zeros_like(stims_ave)
    for idx, s in enumerate(stims_ave):
        if s in sharedix:
            tr_idx[idx] = 0
        else:
            tr_idx[idx] = 1

    # Load Stable Diffusion Model
    config = '../stable-diffusion_v1/configs/stable-diffusion/v1-inference.yaml'
    ckpt = '../stable-diffusion_v1/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt'
    ckpt = '../stable-diffusion_v1/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt'

    config = OmegaConf.load(f"{config}")
    torch.cuda.set_device(gpu)
    model = load_model_from_config(config, f"{ckpt}", gpu)

    n_samples = 1
    ddim_steps = 50
    ddim_eta = 0.0
    strength = 0.8
    scale = 5.0
    n_iter = 5
    precision = 'autocast'
    precision_scope = autocast if precision == "autocast" else nullcontext
    batch_size = n_samples
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 解码结果输出路径
    outdir = f'../../Brain-Decoded/{subject}/image-{method}/'
    # outdir = f'../../Brain-Decoded/cross-subj/{subject}/image-{method}/'
    os.makedirs(outdir, exist_ok=True)

    precision = 'autocast'
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    for imgidx in range(imgids[0],imgids[1]):
        print(f"\n ...now decoding img{imgidx}... \n")
        
        # Load z (Image)
        imgidx_te = np.where(tr_idx==0)[0][imgidx] # Extract test image index
        idx73k= stims_ave[imgidx_te]
        Image.fromarray(np.squeeze(sdataset[idx73k,:,:,:]).astype(np.uint8)).save(
            os.path.join(outdir, f"{imgidx:05}_org.png"))    
        
        if method in ['init','text']:
            roi_latent = 'early'
            init_latent = np.load(f'../data/decoded/{subject}/{subject}_{roi_latent}_brain_embs_init_latent.npy')
            imgarr = torch.Tensor(init_latent[imgidx,:].reshape(4,40,40)).unsqueeze(0).to('cuda')
        
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
        
        elif method == 'gan':
            ganpath = f'{gandir}/recon_image_normalized-VGG19-fc8-{subject}-streams-{imgidx:06}.tiff'
            im = Image.open(ganpath).resize((512,512))
            im = np.array(im)
        
        init_image = load_img_from_arr(im).to('cuda')
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        
        # Load c (Semantics)
        if method == 'init':
            roi_c = 'ventral'
            c_embs = np.load(f'../data/decoded/{subject}/{subject}_{roi_c}_brain_embs_c.npy')
            carr = c_embs[imgidx,:].reshape(77,768)
            c = torch.Tensor(carr).unsqueeze(0).to('cuda')
        elif method in ['text','gan']:
            captions = pd.read_csv(f'{captdir}/captions_brain.csv', sep='\t',header=None)
            c = model.get_learned_conditioning(captions.iloc[imgidx][0]).to('cuda')

        # Generate image from Z (image) + C (semantics)
        base_count = 0
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    for n in range(n_iter):
                        uc = model.get_learned_conditioning(batch_size * [""])
        
                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc,)
        
                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        
                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(outdir, f"{imgidx:05}_{base_count:03}.png"))    
                        base_count += 1



if __name__ == "__main__":
    main()
