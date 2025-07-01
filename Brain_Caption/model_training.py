'''
模型训练模块
功能：加载预处理的数据，训练大脑到图像嵌入的映射模型
'''

import os
from os.path import join as opj
import torch
import tqdm
from sklearn.linear_model import RidgeCV
import pickle

def load_preprocessed_data(subj):
    """加载预处理的数据"""
    base_dir = f"models/{subj}"
    train_fmri = torch.load(f"{base_dir}/train_fmri.pt")
    train_clip_img_embeds = torch.load(f"{base_dir}/train_clip_img_embeds.pt")
    test_fmri = torch.load(f"{base_dir}/test_fmri.pt")
    
    return train_fmri, train_clip_img_embeds, test_fmri

def normalize_data(train_fmri, test_fmri):
    """标准化fMRI数据"""
    train_fmri_mean = torch.mean(train_fmri, axis=0)
    train_fmri_std = torch.std(train_fmri, axis=0)
    
    train_fmri_norm = (train_fmri - train_fmri_mean) / train_fmri_std
    test_fmri_norm = (test_fmri - train_fmri_mean) / train_fmri_std
    
    return train_fmri_norm, test_fmri_norm

def train_and_save_models(train_fmri_norm, train_clip_img_embeds, subj):
    """训练并保存映射模型"""
    from joblib import Parallel, delayed
    import multiprocessing
    
    max_len_img = 197
    alphas = 5e4
    
    # 预先转换数据
    train_fmri_numpy = train_fmri_norm.numpy()
    train_clip_numpy = train_clip_img_embeds.numpy()
    
    # 修改训练函数，返回索引和模型的元组
    def train_single_model(i):
        m = RidgeCV(alphas)
        m.fit(train_fmri_numpy, train_clip_numpy[:,i,:])
        return (i, m)  # 返回索引和模型的元组
    
    # 并行训练模型
    n_jobs = multiprocessing.cpu_count() - 1  # 留一个CPU核心给系统
    print(f"Training brain-to-image embedding models using {n_jobs} cores...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_single_model)(i) for i in tqdm.tqdm(range(max_len_img))
    )
    
    # 按索引排序结果
    sorted_results = sorted(results, key=lambda x: x[0])
    brain_to_img_emb = [model for _, model in sorted_results]  # 提取排序后的模型
    
    # 验证结果
    print(f"验证训练结果: 获得了 {len(brain_to_img_emb)} 个模型 (应该是 {max_len_img} 个)")
    assert len(brain_to_img_emb) == max_len_img, f"模型数量不匹配: {len(brain_to_img_emb)} != {max_len_img}"
    
    # 验证每个模型都是有效的 RidgeCV 实例
    for i, model in enumerate(brain_to_img_emb):
        if not isinstance(model, RidgeCV):
            raise ValueError(f"第 {i} 个模型类型错误: {type(model)}")
    
    # 保存模型
    save_dir = f"models/{subj}/decoding"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Saving brain-to-image embedding models...")
    for i in range(max_len_img):
        filename = f'brain_to_img_emb_ridge_{i}.sav'
        with open(opj(save_dir, filename), 'wb') as f:
            pickle.dump(brain_to_img_emb[i], f)
    
    # 保存后验证
    print("验证保存的模型文件...")
    for i in range(max_len_img):
        filename = f'brain_to_img_emb_ridge_{i}.sav'
        filepath = opj(save_dir, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"模型文件丢失: {filepath}")
        # 可选：验证文件大小不为0
        if os.path.getsize(filepath) == 0:
            raise ValueError(f"模型文件为空: {filepath}")
    
    return brain_to_img_emb

def compute_and_save_statistics(train_clip_img_embeds, brain_to_img_emb, train_fmri_norm, subj):
    """计算并保存数据统计信息"""
    max_len_img = 197
    
    # 计算训练嵌入预测
    img_emb_train = []
    for i in tqdm.tqdm(range(max_len_img)):
        emb = torch.tensor(brain_to_img_emb[i].predict(train_fmri_norm.numpy()))
        img_emb_train.append(emb)
    img_emb_train = torch.stack(img_emb_train, 1)
    
    # 计算统计量
    train_clip_img_embeds_mean = train_clip_img_embeds.mean(0)
    train_clip_img_embeds_std = train_clip_img_embeds.std(0)
    pred_clip_img_embeds_mean = img_emb_train.mean(0)
    pred_clip_img_embeds_std = img_emb_train.std(0)
    
    # 保存统计量
    save_dir = f"models/{subj}"
    torch.save(train_clip_img_embeds_mean, opj(save_dir, "train_clip_img_embeds_mean.pt"))
    torch.save(train_clip_img_embeds_std, opj(save_dir, "train_clip_img_embeds_std.pt"))
    torch.save(pred_clip_img_embeds_mean, opj(save_dir, "pred_clip_img_embeds_mean.pt"))
    torch.save(pred_clip_img_embeds_std, opj(save_dir, "pred_clip_img_embeds_std.pt"))

def main():
    subj = "subj02"
    
    # 加载预处理的数据
    print("Loading preprocessed data...")
    train_fmri, train_clip_img_embeds, test_fmri = load_preprocessed_data(subj)
    
    # 数据标准化
    print("Normalizing data...")
    train_fmri_norm, test_fmri_norm = normalize_data(train_fmri, test_fmri)
    
    # 训练并保存模型
    brain_to_img_emb = train_and_save_models(train_fmri_norm, train_clip_img_embeds, subj)
    
    # 计算并保存统计信息
    print("Computing and saving statistics...")
    compute_and_save_statistics(train_clip_img_embeds, brain_to_img_emb, train_fmri_norm, subj)
    
    print("Model training completed!")

if __name__ == "__main__":
    main() 
