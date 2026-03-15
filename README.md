# Reg-TTR, Test-Time Refinement for Fast, Robust and Accurate Image Registration


![Pytorch](https://img.shields.io/badge/Implemented%20in-Pytorch-red.svg) <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2601.19114-b31b1b.svg)](https://doi.org/10.48550/arXiv.2601.19114)


This repository hosts the official PyTorch implementation of "REG-TTR, Test-Time Refinement for Fast, Robust and Accurate Image Registration". Reg-TTR is an efficient test-time refinement framework for medical image registration that synergizes the complementary strengths of deep learning and conventional registration techniques. It can achieve state-of-the-art registration accuracy on diverse medical image registration tasks while maintaining fast inference speeds close to mainstream deep learning-based methods, with only 21% additional inference time (0.56s) incurred by the refinement process. We have demonstrated its efficacy in unsupervised inter-subject [Abdomen CT](https://drive.usercontent.google.com/download?id=1aWyS_mQ5n7X2bTk9etHrn5di2-EZEzyO&export=download&authuser=0) registration and unsupervised intra-subject cardiac MR ([ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) dataset) registration, and verified its generalizability in boosting the performance of various pre-trained registration models including registration foundation models and task-specific specialized models.

<p align="center">
    <img src="./figs/Reg-TTR.jpg" width="600"/>
</p>


## Highlights

<p align="center">
    <img src="./figs/visualization.jpg" width="600"/>
</p>

<p align="center">
    <img src="./figs/ACDC.jpg" width="600"/>
</p>

**Quantitative comparison on the Abdomen CT dataset.** ↑ higher is better, ↓ lower is better. *uniGradICON\* denotes utilizing uniGradICON's own instance optimization.*

| Model | Type | Dice (%) ↑ | HD95 (mm) ↓ | SDlogJ ↓ | Time (s) |
|---|---|---|---|---|---|
| Initial | - | 30.86 | 11.95 | 0.00 | - |
| VoxelMorph | Semi | 47.05 | 23.08 | 0.13 | < 1.0 |
| FourierNet | Semi | 42.80 | 22.95 | 0.13 | < 1.0 |
| CorrMLP | Semi | 56.58 | 20.40 | 0.16 | < 1.0 |
| VoxelMorph | Un | 41.90 | 25.97 | 0.12 | < 1.0 |
| FourierNet | Un | 41.83 | 25.25 | 0.11 | < 1.0 |
| CorrMLP | Un | 51.01 | 22.80 | 0.13 | < 1.0 |
| ConvexAdam | Un | 50.23 | 22.60 | 0.13 | 7.0 |
| uniGradICON | Un | 53.33 | 20.20 | 0.13 | 2.64 |
| uniGradICON* | Un | 53.99 | 19.94 | 0.17 | 32.48 |
| **Reg-TTR (Ours)** | Un | **56.81** | 20.15 | 0.17 | 3.20 |
| w/o $L_{ssim}$ | Un | 56.45 | 20.10 | 0.15 | 3.18 |

Reg-TTR is a novel efficient test-time refinement framework for medical image registration, which can be incorporated with various pre-trained registration models like registration foundation models ([uniGradICON](https://github.com/uncbiag/uniGradICON)), task-specific specialized models ([VoxelMorph](https://github.com/voxelmorph/voxelmorph), [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)) and [RDP](https://github.com/ZAX130/RDP), [MemWarp](https://github.com/tinymilky/Mem-Warp) to achieve superior registration accuracy while maintaining fast inference speed.


## Datasets
Pretrained Weights for uniGradICON 
The pre-trained weights of the uniGradICON registration foundation model used in this project are obtained directly from the official release of the uniGradICON model. You can find the uniGradICON model weights [here](https://github.com/uncbiag/uniGradICON/releases).

The datasets used are **[Abdomen CT](https://drive.usercontent.google.com/download?id=1aWyS_mQ5n7X2bTk9etHrn5di2-EZEzyO&export=download&authuser=0)** and **[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)**.

## Usage
Run the following commands in the `./src` folder to reproduce the results:

```plain
python testabdomen.py -m UniGradICON -d abdomenreg -bs 1 --num_classes 14
python testACDC.py -m UniGradICON -d acdcreg -bs 1 --num_classes 4
```

- `-m UniGradICON`: Model name, set to "UniGradICON".
- `-d abdomenreg`: Dataset used, specifically "abdomenreg".
- `-bs 1`: Batch size, defined as 1.

## Citation
If our work has influenced or contributed to your research, please kindly acknowledge it by citing:
```
@article{chen2026reg,
  title={Reg-TTR, Test-Time Refinement for Fast, Robust and Accurate Image Registration},
  author={Chen, Lin and He, Yue and Zhang, Fengting and Wang, Yaonan and Lin, Fengming and Chen, Xiang and Liu, Min},
  journal={arXiv preprint arXiv:2601.19114},
  year={2026}
}
```
