import argparse, os
import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
from himalaya.scoring import correlation_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=str,
        default='',
        help="Target variable",
    )

    # parser.add_argument(
    #     "--roi",
    #     type=str,
    #     nargs='*',
    #     help="roi names",
    # )

    opt = parser.parse_args()
    target = opt.target
    
    roi = ['early','midlateral','midparietal','midventral','lateral','parietal','ventral'] 
    # roi = opt.roi

    subject='subj07'
    
    # 路径设置
    mridir = f'../Brain-Diffusion/data/fmri/{subject}'
    featdir = f'../data/nsdfeat/{subject}_feat'
    
    savedir = f'../data/decoded/{subject}'
    os.makedirs(savedir, exist_ok=True)
    
    # 超参数选择，正则化系数
    alpha = 5e4
    
    '''算法流程'''
    # 创建岭回归
    ridge = Ridge(alpha=alpha)

    # if isinstance(alpha, (int, float)):
    #     ridge = Ridge(alpha=alpha)
    # else:
    #     # 如果传入的是数组，保留RidgeCV
    #     ridge = RidgeCV(alphas=alpha)
    
    # 创建预处理流程
    preprocess_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
    )
    
    # 预处理+岭回归 流程pipeline
    pipeline = make_pipeline(
        preprocess_pipeline,
        ridge,
    ) 
    
    '''加载训练和测试数据'''
    # 使用并行加载数据
    def load_roi_data(subject, croi, mridir):
        print(f'Loading {subject} {croi} data...')
        cX = np.load(f'{mridir}/{subject}_{croi}_betas_tr.npy').astype("float32")
        cX_te = np.load(f'{mridir}/{subject}_{croi}_betas_ave_te.npy').astype("float32")
        return cX, cX_te
    
    # 并行加载所有ROI数据
    results = Parallel(n_jobs=-1)(delayed(load_roi_data)(subject, croi, mridir) for croi in roi)
    X_list, X_te_list = zip(*results)
    
    # 使用np.concatenate代替np.hstack (性能更好)
    X = np.concatenate(X_list, axis=1)
    X_te = np.concatenate(X_te_list, axis=1)
    
    # 添加内存优化：释放不再需要的变量
    del X_list, X_te_list
    
    # 目标数据（图像特征）
    print(f'Loading {subject} {target} data...')
    Y = np.load(f'{featdir}/{subject}_each_{target}_tr.npy').astype("float32").reshape([X.shape[0],-1])
    Y_te = np.load(f'{featdir}/{subject}_ave_{target}_te.npy').astype("float32").reshape([X_te.shape[0],-1])
    
    # 训练回归拟合模型
    print(f'Now making decoding model for... {subject}:  {roi}, {target}')
    pipeline.fit(X, Y)
    
    # 模型预测和评估
    print(f'predicting for... {subject}:  {roi}, {target}')
    brain_embs = pipeline.predict(X_te)
    rs = correlation_score(Y_te.T,brain_embs.T)
    print(f'Prediction accuracy is: {np.mean(rs):3.3}')
    
    # if isinstance(ridge, RidgeCV):
    #     best_alpha = pipeline.named_steps['ridgecv'].best_alphas_
    #     print("Best alpha parameter:", best_alpha)
    
    # 保存预测结果
    np.save(f'{savedir}/{subject}_{"_".join(roi)}_brain_embs_{target}.npy',brain_embs)


if __name__ == "__main__":
    main()
