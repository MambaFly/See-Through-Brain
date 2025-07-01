'''
评估指标计算模块
包含METEOR、Sentence Transformer和CLIP三种相似度计算方法
'''
import evaluate
from sentence_transformers import SentenceTransformer, util
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
import pickle
import torch
import numpy as np


def load_captions(subj):
    '''
    加载三种类型的字幕数据
    Args:
        subj: 被试编号
    Returns:
        test_captions: 人工标注字幕
        captions_from_images: 图像生成字幕
        captions_from_brain: 大脑信号生成字幕
    '''
    # 人工标注字幕(真实标签)
    with open(f"models/{subj}/test_captions.sav", "rb") as f:
        test_captions = pickle.load(f)
    # caps = np.load('./processed_data/subj01/best_test_captions.npy', allow_pickle=True)
    # test_captions = caps.tolist()
    
    print(test_captions[0])

    # 图像字幕
    with open(f"decoded_captions/{subj}/image_captions.sav", "rb") as f:
        captions_from_images = pickle.load(f)

    # 大脑字幕
    with open(f"decoded_captions/{subj}/brain_captions.sav", "rb") as f:
        captions_from_brain = pickle.load(f)
        
    return test_captions, captions_from_images, captions_from_brain

def compute_meteor_scores(test_captions, captions_from_images, captions_from_brain):
    '''
    计算METEOR评分
    '''
    meteor = evaluate.load('meteor')
    
    meteor_img_ref = meteor.compute(predictions=captions_from_images, references=test_captions)
    meteor_brain_img = meteor.compute(predictions=captions_from_brain, references=captions_from_images)
    meteor_brain_ref = meteor.compute(predictions=captions_from_brain, references=test_captions)
    
    return meteor_img_ref['meteor'], meteor_brain_img['meteor'], meteor_brain_ref['meteor']

def compute_sentence_transformer_similarity(test_captions, captions_from_images, captions_from_brain):
    '''
    计算Sentence Transformer相似度
    '''
    sentence_model = SentenceTransformer('./sentence-transformers/all-MiniLM-L6-v2')
    
    with torch.no_grad():
        embedding_brain = sentence_model.encode(captions_from_brain, convert_to_tensor=True)
        embedding_captions = sentence_model.encode(test_captions, convert_to_tensor=True)
        embedding_images = sentence_model.encode(captions_from_images, convert_to_tensor=True)

        ss_sim_brain_img = util.pytorch_cos_sim(embedding_brain, embedding_images).cpu()
        ss_sim_img_cap = util.pytorch_cos_sim(embedding_images, embedding_captions).cpu()
        ss_sim_brain_cap = util.pytorch_cos_sim(embedding_brain, embedding_captions).cpu()
        
    return ss_sim_img_cap.diag().mean(), ss_sim_brain_img.diag().mean(), ss_sim_brain_cap.diag().mean()

def compute_clip_similarity(test_captions, captions_from_images, captions_from_brain):
    '''
    计算CLIP相似度
    '''
    model_clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("./openai/clip-vit-base-patch32")
    
    with torch.no_grad():
        input_ids = tokenizer(captions_from_brain, return_tensors="pt", padding=True)
        embedding_brain = model_clip.get_text_features(**input_ids)

        input_ids = tokenizer(test_captions, return_tensors="pt", padding=True)
        embedding_captions = model_clip.get_text_features(**input_ids)

        input_ids = tokenizer(captions_from_images, return_tensors="pt", padding=True)
        embedding_images = model_clip.get_text_features(**input_ids)

    clip_sim_brain_img = util.pytorch_cos_sim(embedding_brain, embedding_images).cpu()
    clip_sim_img_cap = util.pytorch_cos_sim(embedding_images, embedding_captions).cpu()
    clip_sim_brain_cap = util.pytorch_cos_sim(embedding_brain, embedding_captions).cpu()
    
    return clip_sim_img_cap.diag().mean(), clip_sim_brain_img.diag().mean(), clip_sim_brain_cap.diag().mean()

def main():
    # 初始化参数
    subj = 'subj07'
    
    # 加载数据
    test_captions, captions_from_images, captions_from_brain = load_captions(subj)
    
    # 计算METEOR分数
    meteor_img_ref, meteor_brain_img, meteor_brain_ref = compute_meteor_scores(
        test_captions, captions_from_images, captions_from_brain
    )
    print(f"[GROUND] METEOR GIT from images vs captions: {meteor_img_ref}")
    print(f"[ABSOLUTE] METEOR GIT from brain vs images: {meteor_brain_img}")
    print(f"[RELATIVE] METEOR GIT from brain vs captions: {meteor_brain_ref}")
    
    # 计算Sentence Transformer相似度
    st_img_cap, st_brain_img, st_brain_cap = compute_sentence_transformer_similarity(
        test_captions, captions_from_images, captions_from_brain
    )
    print(f"[GROUND] Sentence Transformer Similarity GIT from image vs human: {st_img_cap}")
    print(f"[ABSOLUTE] Sentence Transformer Similarity GIT from brain vs image: {st_brain_img}")
    print(f"[RELATIVE] Sentence Transformer Similarity GIT from brain vs human: {st_brain_cap}")
    
    # 计算CLIP相似度
    clip_img_cap, clip_brain_img, clip_brain_cap = compute_clip_similarity(
        test_captions, captions_from_images, captions_from_brain
    )
    print(f"[GROUND] CLIP Similarity GIT from image vs human: {clip_img_cap}")
    print(f"[ABSOLUTE] CLIP Similarity GIT from brain vs image: {clip_brain_img}")
    print(f"[RELATIVE] CLIP Similarity GIT from brain vs human: {clip_brain_cap}")

if __name__ == "__main__":
    main() 