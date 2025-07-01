import argparse, os
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from nsd_access import NSDAccess
from PIL import Image
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


'''图像转换预处理''' 
def load_img_from_arr(img_arr,resolution):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = resolution, resolution
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

'''模型和参数的加载'''
def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)  # 实例化模型
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda(f"cuda:{gpu}")  # 加载到gpu
    model.eval()
    return model
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--imgidx",
        required=True,
        nargs="*",
        type=int,
        help="start and end imgs"
    )
    # 参数设置
    seed = 42
    seed_everything(seed)
    opt = parser.parse_args()
    imgidx = opt.imgidx
    gpu = 0
    
    # ldm超参数
    resolution = 320  #图像分辨率
    batch_size = 1  #批量大小
    ddim_steps = 50  #扩散步数
    ddim_eta = 0.0 #噪声控制
    strength = 0.8  #强度
    scale = 5.0  #缩放
    
    # 访问数据和模型
    nsda = NSDAccess('../data/nsd/')
    config = '../stable-diffusion_v1/configs/stable-diffusion/v1-inference.yaml'
    ckpt = '../stable-diffusion_v1/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt'
    ckpt = '../stable-diffusion_v1/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt'
    config = OmegaConf.load(f"{config}")
    torch.cuda.set_device(gpu)
    
    # 输出路径
    os.makedirs(f'../data/nsdfeat/init_latent/', exist_ok=True)
    os.makedirs(f'../data/nsdfeat/c/', exist_ok=True)
    
    '''模型加载''' 
    precision = 'autocast'
    precision_scope = autocast if precision == "autocast" else nullcontext
    model = load_model_from_config(config, f"{ckpt}", gpu)  # 加载模型
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)  # 设置DDIM采样器
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")
    
    '''主流程-图像编码成embeddings'''
    for s in range(imgidx[0],imgidx[1]):
        print(f"Now processing image {s:06}")
        prompt = []
        prompts = nsda.read_image_coco_info([s],info_type='captions')
        for p in prompts:
            prompt.append(p['caption'])    
        
        img = nsda.read_images(s)
        # print(img.shape)
        init_image = load_img_from_arr(img,resolution).to(device)
        # print(init_image.shape)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
    
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    # 无条件
                    uc = model.get_learned_conditioning(batch_size * [""])
                    # prompt条件
                    c = model.get_learned_conditioning(prompt).mean(axis=0).unsqueeze(0)
    
                    # encode (scaled latent)
                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                    # decode it
                    samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,)
    
        # 初始潜在表征
        init_latent = init_latent.cpu().detach().numpy().flatten()
        # 条件引导信息
        c = c.cpu().detach().numpy().flatten()
        np.save(f'../data/nsdfeat/init_latent/{s:06}.npy',init_latent)
        np.save(f'../data/nsdfeat/c/{s:06}.npy',c)

if __name__ == "__main__":
    main()

