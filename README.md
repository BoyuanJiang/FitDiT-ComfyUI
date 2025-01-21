# FitDiT: Advancing the Authentic Garment Details for High-fidelity Virtual Try-on

<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://arxiv.org/abs/2411.10499" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2411.10499-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href="https://github.com/BoyuanJiang/FitDiT" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <a href="https://huggingface.co/spaces/BoyuanJiang/FitDiT" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Space-ZeroGPU-orange?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
  <a href="http://demo.fitdit.byjiang.com/" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
  <a href='https://huggingface.co/BoyuanJiang/FitDiT' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Hugging Face-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'>
  </a>
  <a href='https://byjiang.com/FitDiT/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
  <a href="https://raw.githubusercontent.com/BoyuanJiang/FitDiT/refs/heads/main/LICENSE" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
  </a>
</div>

**FitDiT** is designed for high-fidelity virtual try-on using Diffusion Transformers (DiT).
<div align="center">
  <img src="resource/img/teaser.jpg" width="100%" height="100%"/>
</div>


## Updates
- **`2025/1/16`**: We provide the ComfyUI version of FitDiT, you can use FitDiT in ComfyUI now.


## Installation
### Direct Download
Download or clone the repo of [FitDiT-ComfyUI branch](https://github.com/BoyuanJiang/FitDiT/tree/FitDiT-ComfyUI) and place it in the `ComfyUI/custom_nodes/` directory, you can follow the following steps:

1. goto `ComfyUI/custom_nodes` dir in terminal(cmd)
2. `git clone https://github.com/BoyuanJiang/FitDiT.git -b FitDiT-ComfyUI FitDiT`
3. Restart ComfyUI

### ComfyUI-Manager
You can also use [ComfyUI-Manager](https://github.com/lllyasviel/ComfyUI-Manager) to install FitDiT by searching `FitDiT[official]` in the ComfyUI-Manager.
![ComfyUI-Manager](resource/img/ComfyUI-Manager.png)

### Environment
FItDiT was tested under the following environment, but other versions should also work. You can first use your own existing environment.
- torch==2.4.0
- torchvision==0.19.0 
- accelerate==0.31.0
- diffusers==0.31.0
- transformers==4.39.3
- numpy==1.23.0
- scikit-image==0.24.0
- huggingface_hub==0.26.5
- onnxruntime==1.20.1
- opencv-python
- matplotlib==3.8.3
- einops==0.7.0

## Download model
Download the [**FitDiT model**](https://huggingface.co/BoyuanJiang/FitDiT) and place it in the `ComfyUI/models/FitDiT_models` directory, the  [**clip-vit-large-patch14**](https://huggingface.co/openai/clip-vit-large-patch14) and [**CLIP-ViT-bigG-14**](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) and place them in the `ComfyUI/models/clip` directory.

You can download the model with the following command:
```
pip install -U huggingface_hub
python download_model.py --dir /path/to/ComfyUI/
```

## Example workflows
[fitdit_workflow.json](fitdit_workflow.json) is the example workflow of FitDiT in ComfyUI. If you have less GPU memory, you can set `with_offload` or `with_aggressive_offload` to True. Set `with_offload` to True with **moderate gpu memroty, moderate inference time**. Set `with_aggressive_offload` to True with **lowest gpu memroty, longest inference time**.
![workflow](resource/img/workflow.png)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=BoyuanJiang/FitDiT&type=Date)](https://star-history.com/#BoyuanJiang/FitDiT&Date)

## Contact
This model can only be used **for non-commercial use**. For commercial use, please visit [Tencent Cloud](https://cloud.tencent.com/document/product/1668/108532) for support.


## Citation
If you find our work helpful for your research, please consider citing our work.
```
@misc{jiang2024fitditadvancingauthenticgarment,
      title={FitDiT: Advancing the Authentic Garment Details for High-fidelity Virtual Try-on}, 
      author={Boyuan Jiang and Xiaobin Hu and Donghao Luo and Qingdong He and Chengming Xu and Jinlong Peng and Jiangning Zhang and Chengjie Wang and Yunsheng Wu and Yanwei Fu},
      year={2024},
      eprint={2411.10499},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.10499}, 
}
```

