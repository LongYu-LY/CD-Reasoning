<div align="center">
  <h1>[TIP2025]Harnessing Multi-modal Large Language Models for Measuring and Interpreting Color Differences</h1>

<!--  <div style="width: 50%; text-align: center; margin:auto;">
    <img style="width: 50%" src="fig/cd-reasoning-teaser.png">
  </div> -->

  <a href="https://github.com/LongYu-LY/CD-Reasoning">Official Implementation</a>

<div>
    <a href="https://github.com/SMBU-MM" target="_blank">Zhihua Wang</a><sup>1</sup>,
    <a href="https://github.com/LongYu-LY" target="_blank">Yu Long</a><sup>2</sup>,
    <a href="[Qiuping's profile]" target="_blank">Qiuping Jiang</a><sup>3</sup>,
    <a href="[Chao's profile]" target="_blank">Chao Huang</a><sup>4*</sup>,
    <a href="[Xiaochun's profile]" target="_blank">Xiaochun Cao</a><sup>4</sup>
</div>
  
  <div>
    <sup>1</sup>Shenzhen MSU-BIT University,
    <sup>2</sup>Beijing Institute of Technology,
    <sup>3</sup>Ningbo University,
    <sup>4</sup>Sun Yat-sen University
  </div>

  <div>
    <sup>*</sup>Corresponding author
  </div>

  <div>
    <a href="https://ieeexplore.ieee.org/document/10820056"><strong>Paper</strong></a> | 
    <a href="https://github.com/LongYu-LY/CD-Reasoning"><strong>Code</strong></a> | 
  </div>
</div>

## Abstract
We present CD-Reasoning, a novel multimodal method for measuring and interpreting perceptual color differences (CDs) between images. Unlike traditional CD metrics that only provide numerical scores, our approach leverages Multimodal Large Language Models (MLLMs) to deliver both quantitative assessments and human-like qualitative explanations. We introduce the M-SPCD dataset, extending the existing SPCD dataset with over 10,000 expert annotations across seven key color attributes. Extensive experiments demonstrate that CD-Reasoning outperforms state-of-the-art CD metrics in scoring accuracy while providing interpretable reasoning about color differences.

## Key Features
- **Multimodal CD Assessment**: Combines numerical scoring with natural language explanations
- **Comprehensive Dataset**: M-SPCD with 30,000 image pairs and detailed textual descriptions
- **State-of-the-art Performance**: Superior to existing CD metrics in both scoring and interpretation
- **Robust Architecture**: Built on CLIP ViT-L/14 and LLaMA-2-7B foundation models

## Installation
```shell
git clone https://github.com/LongYu-LY/CD-Reasoning.git
cd CD-Reasoning
pip install -e .
```

For training, you need to further install additional dependencies as follows:

```shell
pip install -e ".[train]"
pip install flash_attn --no-build-isolation
```

## Train
Full Fine-tuning Training
```shell
sh scripts/finetune.sh
```

Lora Fine-tuning Training
```shell
sh scripts/finetune_lora.sh

```

## Evaluation
- Color Differenct Measure
```shell
python cdreasoning/evaluate/cd_score.py --model-path cdreasoning-checkpoint --device cuda:0
```
- Color Differenct Description

```shell
python cdreasoning/evaluate/cd_description.py --model-path cdreasoning-checkpoint --device cuda:0
```

## Model Checkpoints and SPCD Dataset

Pretrained model checkpoints are available for download:

- [Baidu Netdisk](https://pan.baidu.com/s/1VK7MGXd1c0_vMp3YAcABQA) (Extraction code: 6s5d)

SPCD are available for download:

- [Baidu Netdisk](https://pan.baidu.com/share/init?surl=8bzu-qhpMW3PqLTlVdoZRQ&pwd=txeh) (Extraction code: txeh)

  
## Citation

```bibtex
@article{wang2024cdreasoning,
  title={Harnessing Multi-modal Large Language Models for Measuring and Interpreting Color Differences},
  author={Wang, Zhihua and Long, Yu and Jiang, Qiuping and Huang, Chao and Cao, Xiaochun},
  journal={IEEE Transactions on Image Processing},
  year={2024}
}
@article{wu2023qalign,
  title={Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels},
  author={Wu, Haoning and Zhang, Zicheng and Zhang, Weixia and Chen, Chaofeng and Li, Chunyi and Liao, Liang and Wang, Annan and Zhang, Erli and Sun, Wenxiu and Yan, Qiong and Min, Xiongkuo and Zhai, Guangtai and Lin, Weisi},
  journal={arXiv preprint arXiv:2312.17090},
  year={2023},
  institution={Nanyang Technological University and Shanghai Jiao Tong University and Sensetime Research},
  note={Equal Contribution by Wu, Haoning and Zhang, Zicheng. Project Lead by Wu, Haoning. Corresponding Authors: Zhai, Guangtai and Lin, Weisi.}
}
```
This repo is based on [Q-Align](https://github.com/Q-Future/Q-Align) [ICML 2024]
