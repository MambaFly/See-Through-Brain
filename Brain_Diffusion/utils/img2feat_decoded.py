import argparse, os, sys, glob
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import torchvision
from torchvision import transforms
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        required=True,
        type=str,
        help="decoded image directory",
    )

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
    #     help="init or text or text-depth",
    # )

    # # Parameters
    opt = parser.parse_args()
    dir = opt.dir
    subject=opt.subject
    # method = opt.method
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    # imglist = sorted(glob.glob(f'../../Brain-Decoded/{subject}/image-{method}/*'))
    # outdir = f'../../Brain-Identification/{subject}/{method}/'

    imglist = sorted(glob.glob(f'../../Brain-Decoded/cross-subj/{subject}/{dir}/*'))
    outdir = f'../../Brain-Identification/{subject}/cross-subj/{dir}/'
    os.makedirs(outdir, exist_ok=True)

    # Load Models 
    # Inception V3
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model_inception = torchvision.models.inception_v3(pretrained=True)
    model_inception.eval()
    model_inception.to(device)
    # 使用hook获取特征
    features_inception = {}
    def hook_inception(module, input, output):
        features_inception['flatten'] = output
    model_inception._modules.get('fc').register_forward_hook(hook_inception)

    # AlexNet
    model_alexnet = torchvision.models.alexnet(pretrained=True)
    model_alexnet.eval()
    model_alexnet.to(device)
    # 使用hook获取特征
    features_alexnet = {}
    # def hook_alexnet5(module, input, output):
    #     features_alexnet['features.5'] = output
    def hook_alexnet12(module, input, output):
        features_alexnet['features.12'] = output
    # def hook_alexnet18(module, input, output):
        features_alexnet['classifier.5'] = output
    # model_alexnet._modules.get('features')._modules.get('5').register_forward_hook(hook_alexnet5)
    model_alexnet._modules.get('features')._modules.get('12').register_forward_hook(hook_alexnet12)
    # model_alexnet._modules.get('classifier')._modules.get('5').register_forward_hook(hook_alexnet18)

    # CLIP
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model_clip.to(device)
    processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    print(f"Now processing start >>> ")
    for img in tqdm(imglist):
        imgname = img.split('/')[-1].split('.')[0]
        print(img)
        image = Image.open(img)

        # Inception
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(device)
        with torch.no_grad():
            _ = model_inception(input_batch)
        feat_inception = features_inception['flatten'].cpu().detach().numpy().copy()    

        # AlexNet
        with torch.no_grad():
            _ = model_alexnet(input_batch)
        # feat_alexnet5 = features_alexnet['features.5'].flatten().cpu().detach().numpy().copy()    
        feat_alexnet12 = features_alexnet['features.12'].flatten().cpu().detach().numpy().copy()    
        # feat_alexnet18 = features_alexnet['classifier.5'].flatten().cpu().detach().numpy().copy()    

        # CLIP
        inputs = processor_clip(text="",images=image, return_tensors="pt").to(device)
        outputs = model_clip(**inputs,output_hidden_states=True)
        feat_clip = outputs.image_embeds.cpu().detach().numpy().copy()
        # feat_clip_h6 = outputs.vision_model_output.hidden_states[6].flatten().cpu().detach().numpy().copy()
        # feat_clip_h12 = outputs.vision_model_output.hidden_states[12].flatten().cpu().detach().numpy().copy()

        # SAVE
        fname = f'{outdir}/{imgname}'
        np.save(f'{fname}_inception.npy',feat_inception)
        # np.save(f'{fname}_alexnet5.npy',feat_alexnet5)
        np.save(f'{fname}_alexnet12.npy',feat_alexnet12)
        # np.save(f'{fname}_alexnet18.npy',feat_alexnet18)
        np.save(f'{fname}_clip.npy',feat_clip)
        # np.save(f'{fname}_clip_h6.npy',feat_clip_h6)
        # np.save(f'{fname}_clip_h12.npy',feat_clip_h12)

if __name__ == "__main__":
    main()
