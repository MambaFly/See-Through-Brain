import os
#os.system('ls -l')

'''
nsd的相关数据下载
'''

# Download Experiment Infos
# 实验信息数据（实验设计，刺激信息）
os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat nsddata/experiments/nsd/')
os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.pkl nsddata/experiments/nsd/')

# Download Stimuli
# nsd刺激数据
os.system('aws s3 cp s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 nsddata_stimuli/stimuli/nsd/')

# Download Betas
# 受试者的大脑响应数据（betas）
# suj01 session01 beta值:表示受试者01在第1次session的刺激任务中，大脑各个体素的响应值，4维张量，750 x 83 x 104 x 81，表示750次刺激实验中，每个体素的响应强度（用广义线性模型GLM分析时的系数来指代）
for sub in [1,2,5,7]:
    for sess in range(1,41):
        os.system('aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{:02d}.nii.gz nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub,sess,sub))

# Download ROIs
# for sub in [1,2,5,7]:
# 受试者的感兴趣区域（ROI）相关数据
# ROIs：空间定义，描述信息（视觉皮层v1~v4），统计数据
for sub in [1,2,5,7]:
    os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/ nsddata/ppdata/subj{:02d}/func1pt8mm/roi/ --recursive'.format(sub,sub))
