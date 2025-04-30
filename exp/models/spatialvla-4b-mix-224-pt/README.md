---
license: mit
language:
- en
base_model:
- IPEC-COMMUNITY/spatialvla-4b-224-pt
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- VLA
- Foundation Vision-language-action Model
- Generalist Robot Policy
- robotics
---

# SpatialVLA Fine-Tuned on fractal & bridge

This model was produced by fine-tuning the [SpatialVLA model](IPEC-COMMUNITY/spatialvla-4b-224-pt) on the **fractal and bridge dataset**. We made a few modifications to the training dataset to improve final performance (see the
[SpatialVLA paper](https://arxiv.org/abs/2501.15830) for details). This model is only used in our TABLE V for Fine-tuning Ablations in Domain Datasets.

## Model Details

### Model Description

- **Developed by:** The SpatialVLA team consisting of researchers from Shanghai AI Laboratory, ShanghaiTech and TeleAI.
- **Model type:** Vision-language-action (language, image => robot actions)
- **Language(s) (NLP):** en
- **License:** MIT
- **Finetuned from model:** [paligemma2-3b-pt-224](https://huggingface.co/google/paligemma2-3b-pt-224)
- **Pretraining Dataset:** [Open X-Embodiment](https://robotics-transformer-x.github.io/) and [RH20T](https://rh20t.github.io/)
- **Repository:** [https://github.com/SpatialVLA/SpatialVLA](https://github.com/SpatialVLA/SpatialVLA)
- **Paper:** [SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model](https://arxiv.org/abs/2501.15830)
- **Project Page & Videos:** [https://spatialvla.github.io/](https://spatialvla.github.io/)

## Uses

SpatialVLA relies solely on HuggingFace Transformers 🤗, making deployment extremely easy. If your environment supports `transformers >= 4.47.0`, you can directly use the following code to load the model and perform inference. (requires 8.5GB of GPU memory).

### Direct Use

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

model_name_or_path="IPEC-COMMUNITY/spatialvla-4b-224-pt"
processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda()

image = Image.open("example.png").convert("RGB")
prompt = "What action should the robot take to pick the cup?"
inputs = processor(images=[image], text=prompt, return_tensors="pt")
generation_outputs = model.predict_action(inputs)

actions = processor.decode_actions(generation_outputs, unnorm_key="bridge_orig/1.0.0")
print(actions)
```

### Out-of-Scope Use

SpatialVLA models do not zero-shot generalize to new (unseen) robot embodiments, or setups that are not represented in the pretraining mix; in these cases, we suggest collecting a dataset of demonstrations on the desired setup, and fine-tuning SpatialVLA models instead.

## How to Get Hands Dirty with the Model

If you want to use the model for fine-tuning or pre-training, you need to clone the [official repository](https://github.com/SpatialVLA/SpatialVLA) first.
```bash
git clone https://github.com/SpatialVLA/SpatialVLA.git
```

, then install the required packages and download the model from the Hugging Face model hub. The VLM backbone of SpatialVLA is PaLiGemma2, which requires transformers >= 4.47.0. Hence, create a Python environment with Python >= 3.10.
```bash
conda create -n spatialvla python=3.10
conda activate spatialvla
```

Install packages from `requirements.txt` file. Note that we use a customised `dlimp` to support seed setting for reproducibility. If you catch any problems, please manually install the dlimp form the [dlimp_custom](https://github.com/SpatialVLA/dlimp_custom).

```bash
pip install -r requirements.txt
```
### Train from Scratch

SpatialVLA is pre-trained with 1.1 Million real-robot demonstrations from the OXE and RH20T dataset on a cluster of 64 A100 GPUs for abut 10 days, using a batch size of 2048. You can pre-train the model from scratch using the following command.

```bash
# torchrun
bash scripts/spatialvla_4b_pretrain/torchrun_pretrain.sh

# or in a slurm cluster
bash scripts/spatialvla_4b_pretrain/slurm_pretrain.sh
```

### Fine-tuning

Most of our fine-tuning experiments are conducted using LoRA on 4 or 8 A100 GPUs.
You can use the following scripts for full-parameter or LoRA fine-tuning. For real-world experiments with small datasets, we prefer using LoRA for fine-tuning.

```bash
# full fine-tuning
bash scripts/spatialvla_4b_finetune/finetune_full.sh

# LoRA fine-tuning
bash scripts/spatialvla_4b_finetune/finetune_lora.sh
```

## Evaluation

- SimplerEnv evaluation on Google Robot tasks.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th rowspan="2">Model</th>
      <th colspan="4">Visual Matching</th>
      <th colspan="4">Variant Aggregation</th>
    </tr>
    <tr style="text-align: center;">
      <th>Pick Coke Can</th>
      <th>Move Near</th>
      <th>Open/Close Drawer</th>
      <th>#Average</th>
      <th>Pick Coke Can</th>
      <th>Move Near</th>
      <th>Open/Close Drawer</th>
      <th>#Average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>RT-1 (Begin)</td>
      <td>2.7%</td>
      <td>5.0%</td>
      <td>13.9%</td>
      <td>6.8%</td>
      <td>2.2%</td>
      <td>4.0%</td>
      <td>6.9%</td>
      <td>4.2%</td>
    </tr>
    <tr>
      <td>RT-1 (15%)</td>
      <td>71.0%</td>
      <td>35.4%</td>
      <td>56.5%</td>
      <td>60.2%</td>
      <td>81.3%</td>
      <td>44.6%</td>
      <td>26.7%</td>
      <td>56.2%</td>
    </tr>
    <tr>
      <td>RT-1 (Converged)</td>
      <td>85.7%</td>
      <td>44.2%</td>
      <td>73.0%</td>
      <td>74.6%</td>
      <td>89.8%</td>
      <td>50.0%</td>
      <td>32.3%</td>
      <td>63.3%</td>
    </tr>
    <tr>
      <td>HPT</td>
      <td>56.0%</td>
      <td>60.0%</td>
      <td>24.0%</td>
      <td>46.0%</td>
      <td>--</td>
      <td>--</td>
      <td>31.0%</td>
      <td>45.0%</td>
    </tr>
    <tr>
      <td>TraceVLA</td>
      <td>28.0%</td>
      <td>53.7%</td>
      <td>57.0%</td>
      <td>42.0%</td>
      <td>60.0%</td>
      <td>56.4%</td>
      <td>29.4%</td>
      <td>39.6%</td>
    </tr>
    <tr>
      <td>RT-1-X</td>
      <td>56.7%</td>
      <td>31.7%</td>
      <td>59.7%</td>
      <td>53.4%</td>
      <td>49.0%</td>
      <td>32.3%</td>
      <td>35.3%</td>
      <td>64.3%</td>
    </tr>
    <tr>
      <td>RT-2-X</td>
      <td>78.7%</td>
      <td>77.9%</td>
      <td>25.0%</td>
      <td>60.7%</td>
      <td>82.3%</td>
      <td>79.2%</td>
      <td>--</td>
      <td>--</td>
    </tr>
  <tr>
      <td>Octo-Base</td>
      <td>17.0%</td>
      <td>4.2%</td>
      <td>22.7%</td>
      <td>16.8%</td>
      <td>0.6%</td>
      <td>3.1%</td>
      <td>1.1%</td>
      <td>1.1%</td>
    </tr>
    <tr>
      <td>OpenVLA</td>
      <td>16.3%</td>
      <td>46.2%</td>
      <td>35.6%</td>
      <td>27.7%</td>
      <td>54.5%</td>
      <td>47.7%</td>
      <td>17.7%</td>
      <td>39.8%</td>
    </tr>
    <tr>
      <td>RoboVLM (zero-shot)</td>
      <td>72.7%</td>
      <td>66.3%</td>
      <td>26.8%</td>
      <td>56.3%</td>
      <td>68.3%</td>
      <td>56.0%</td>
      <td>8.5%</td>
      <td>46.3%</td>
    </tr>
    <tr>
      <td>RoboVLM (fine-tuning)</td>
      <td>77.3%</td>
      <td>61.7%</td>
      <td>43.5%</td>
      <td>63.4%</td>
      <td>75.6%</td>
      <td>60.0%</td>
      <td>10.6%</td>
      <td>51.3%</td>
    </tr>
    <tr>
      <td>SpatialVLA (zero-shot)</td>
      <td><b>81.0%</b></td>
      <td><b>69.6%</b></td>
      <td><b>59.3%</b></td>
      <td><b>71.9%</b></td>
      <td><b>89.5%</b></td>
      <td><b>71.7%</b></td>
      <td>36.2%</td>
      <td><b>68.8%</b></td>
    </tr>
    <tr>
      <td>SpatialVLA (fine-tuning)</td>
      <td><b>86.0%</b></td>
      <td><b>77.9%</b></td>
      <td>57.4%</td>
      <td><b>75.1%</b></td>
      <td>88.0%</td>
      <td>72.7%</td>
      <td>41.8%</td>
      <td><b>70.7%</b></td>
    </tr>
  </tbody>
</table>


- SimplerEnv evaluation on WidowX Robot tasks.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th rowspan="2">Model</th>
      <th colspan="2">Put Spoon on Towel</th>
      <th colspan="2">Put Carrot on Plate</th>
      <th colspan="2">Stack Green Block on Yellow Block</th>
      <th colspan="2">Put Eggplant in Yellow Basket</th>
      <th rowspan="2">#Overall Average</th>
    </tr>
    <tr style="text-align: center;">
      <th>Grasp Spoon</th>
      <th>Success</th>
      <th>Grasp Carrot</th>
      <th>Success</th>
      <th>Grasp Green Block</th>
      <th>Success</th>
      <th>Grasp Eggplant</th>
      <th>Success</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>RT-1-X</td>
      <td>16.7%</td>
      <td>0.0%</td>
      <td>20.8%</td>
      <td>4.2%</td>
      <td>8.3%</td>
      <td>0.0%</td>
      <td>0.0%</td>
      <td>0.0%</td>
      <td>1.1%</td>
    </tr>
    <tr>
      <td>Octo-Base</td>
      <td>34.7%</td>
      <td>12.5%</td>
      <td>52.8%</td>
      <td>8.3%</td>
      <td>31.9%</td>
      <td>0.0%</td>
      <td>66.7%</td>
      <td>43.1%</td>
      <td>16.0%</td>
    </tr>
    <tr>
      <td>Octo-Small</td>
      <td>77.8%</td>
      <td>47.2%</td>
      <td>27.8%</td>
      <td>9.7%</td>
      <td>40.3%</td>
      <td>4.2%</td>
      <td>87.5%</td>
      <td>56.9%</td>
      <td>30.0%</td>
    </tr>
    <tr>
      <td>OpenVLA</td>
      <td>4.1%</td>
      <td>0.0%</td>
      <td>33.3%</td>
      <td>0.0%</td>
      <td>12.5%</td>
      <td>0.0%</td>
      <td>8.3%</td>
      <td>4.1%</td>
      <td>1.0%</td>
    </tr>
    <tr>
      <td>RoboVLM (zero-shot)</td>
      <td>37.5%</td>
      <td>20.8%</td>
      <td>33.3%</td>
      <td>25.0%</td>
      <td>8.3%</td>
      <td>8.3%</td>
      <td>0.0%</td>
      <td>0.0%</td>
      <td>13.5%</td>
    </tr>
    <tr>
      <td>RoboVLM (fine-tuning)</td>
      <td>54.2%</td>
      <td>29.2%</td>
      <td>25.0%</td>
      <td>25.0%</td>
      <td>45.8%</td>
      <td>12.5%</td>
      <td>58.3%</td>
      <td>58.3%</td>
      <td>31.3%</td>
    </tr>
    <tr>
      <td>SpatialVLA (zero-shot)</td>
      <td><b>25.0%</b></td>
      <td><b>20.8%</b></td>
      <td><b>41.7%</b></td>
      <td>20.8%</td>
      <td><b>58.3%</b></td>
      <td>25.0%</td>
      <td><b>79.2%</b></td>
      <td>70.8%</td>
      <td><b>34.4%</b></td>
    </tr>
    <tr>
      <td>SpatialVLA (fine-tuning)</td>
      <td><b>20.8%</b></td>
      <td>16.7%</td>
      <td>29.2%</td>
      <td>25.0%</td>
      <td><b>62.5%</b></td>
      <td>29.2%</td>
      <td><b>100.0%</b></td>
      <td><b>100.0%</b></td>
      <td><b>42.7%</b></td>
    </tr>
  </tbody>
</table>

- Zero-shot Robot Control Evaluation on WidowX Robot.

<img src="https://cdn-uploads.huggingface.co/production/uploads/6535045a910b844786a6642f/SUPyXwcdfnWranO04tulL.png" alt="perform">

- Spatial Understanding Capability Evaluation.

<img src="https://cdn-uploads.huggingface.co/production/uploads/6535045a910b844786a6642f/g-EfM-6M7iM9IYryUTwLA.png" alt="perform">


## Citation

**BibTeX:**

```BibTeX
@misc{qu2025spatialvlaexploringspatialrepresentations,
      title={SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model}, 
      author={Delin Qu and Haoming Song and Qizhi Chen and Yuanqi Yao and Xinyi Ye and Yan Ding and Zhigang Wang and JiaYuan Gu and Bin Zhao and Dong Wang and Xuelong Li},
      year={2025},
      eprint={2501.15830},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2501.15830}, 
}
```