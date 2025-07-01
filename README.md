# See Through Brain
The original code of the paper “See Through Brain...”  

# Data Source
NSD data can be freely requested and downloaded at https://naturalscenesdataset.org/   
and the COCO captions are downloadable from https://cocodataset.org/#home, 2017 split.

# Environment Setup
All relevant environment dependencies are stored in the environment.yml file, you can create an environment and activate it with the following command  
```
conda env create -f environment.yml
conda activate brain
```

# Methods
In this study, based on the natural scene dataset and the Stable Diffusion model, we constructed a visual reconstruction process of multimodal fusion by drawing on the hierarchical processing mechanism of the brain's visual system for visual information, as shown here.  
![image](https://github.com/user-attachments/assets/2fd3739e-ad4b-4f12-a6dd-18d3ee4e2c05)


Based on the correspondence between brain visual mechanisms and functional modules, we constructed a multimodal fusion-driven visual reconstruction framework. The framework of this study is guided by the multimodal perception mechanism of the brain, starting from primary visual reconstruction, fusing semantic, depth and other information to realize high-precision reconstruction of visual stimuli. Here is the overall flow of the algorithm.  
![image](https://github.com/user-attachments/assets/f16e4960-0552-43bc-871a-b508a26aed7f)


In order to realize cross-subject visual reconstruction, it is necessary to align the data of other subjects with the baseline subject.  
![image](https://github.com/user-attachments/assets/4f44a019-c220-4f19-80f8-68f80e1c3d88)


# Results
Here are examples of visual reconstruction with multimodal fusion  
![image](https://github.com/user-attachments/assets/05836e0a-a8bd-4ccf-a634-ff2cb0bedb64)

Here are examples of cross-subject visual reconstruction  
![image](https://github.com/user-attachments/assets/5366cbfd-0302-48c4-8300-9eae6de17f29)


# Acknowledgement
Our codebase builds on these repositories. We would like to thank the authors.  
https://github.com/tknapen/nsd_access  
https://github.com/CompVis/stable-diffusion  
https://github.com/Stability-AI/stablediffusion  
https://github.com/ozcelikfu/brain-diffuser  
https://github.com/microsoft/GenerativeImage2Text  
https://github.com/enomodnara/BrainCaptioning  
https://github.com/isl-org/DPT  
https://github.com/yu-takagi/StableDiffusionReconstruction  



