import numpy as np
import scipy.io
from tqdm import tqdm
import argparse
import os

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--featname",
        type=str,
        default='',
        help="Target variable",
    )
    parser.add_argument(
        "--use_stim",
        type=str,
        default='',
        help="ave or each",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    opt = parser.parse_args()
    subject=opt.subject
    use_stim = opt.use_stim
    featname = opt.featname
    # 数据路径
    basedir = '../data/nsdfeat/'
    featdir = f'{basedir}/{featname}/'
    nsd_expdesign = scipy.io.loadmat('../Brain-Diffusion/data/nsd/nsddata/experiments/nsd/nsd_expdesign.mat')
    
    #　保存路径
    savedir = f'{basedir}/{subject}_feat/'
    os.makedirs(savedir, exist_ok=True)
    
    # Note that most of them are 1-base index!
    # This is why I subtract 1
    sharedix = nsd_expdesign['sharedix'] -1 
    print(f'测试集大小：{len(sharedix[0])}')
    if use_stim == 'ave':
        stims = np.load(f'../Brain-Diffusion/data/stim/{subject}/{subject}_stims_ave.npy') # 平均值 30000/3 = 10000 (9000+1000)
    else: # Each
        stims = np.load(f'../Brain-Diffusion/data/stim/{subject}/{subject}_stims.npy')  # 每人30000次刺激

    # 特征数据和索引信息
    feats = []
    tr_idx = np.zeros(len(stims))
    
    # 遍历刺激数据，加载图像特征数据，并更新索引
    for idx, stim in tqdm(enumerate(stims)): 
        # if idx % 1000 == 0:
        #     print(idx,stim)
        if stim in sharedix:  # 测试数据
            tr_idx[idx] = 0
        else:  # 训练数据
            tr_idx[idx] = 1
        # 读取对应的刺激图像的表征数据
        feat = np.load(f'{featdir}/{stim:06}.npy')
        feats.append(feat)
    feats = np.stack(feats)    
    
    
    
    # 训练&测试 图像表征集合
    feats_tr = feats[tr_idx==1,:]
    feats_te = feats[tr_idx==0,:]
    
    # 保存subj的刺激对应 tr_or_te
    np.save(f'../data/stim/{subject}/{subject}_stims_tridx.npy',tr_idx)
    
    # 保存subj的tr和te刺激图像的表征数据
    np.save(f'{savedir}/{subject}_{use_stim}_{featname}_tr.npy',feats_tr)
    np.save(f'{savedir}/{subject}_{use_stim}_{featname}_te.npy',feats_te)

    print(feats_tr.shape)
    print(feats_te.shape)


if __name__ == "__main__":
    main()






