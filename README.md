# Reg-TTR


![Pytorch](https://img.shields.io/badge/Implemented%20in-Pytorch-red.svg) <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2601.19114-b31b1b.svg)](https://doi.org/10.48550/arXiv.2601.19114)

**a registration framework that effectively combines a foundation model with test-time refinement**
*Accepted at ISBI 2026*

This repository hosts the official PyTorch implementation of "REG-TTR, Test-Time Refinement for Fast, Robust and Accurate Image Registration". Reg-TTR is an efficient test-time refinement framework for medical image registration that synergizes the complementary strengths of deep learning and conventional registration techniques. It can achieve state-of-the-art registration accuracy on diverse medical image registration tasks while maintaining fast inference speeds close to mainstream deep learning-based methods, with only 21% additional inference time (0.56s) incurred by the refinement process. We have demonstrated its efficacy in unsupervised inter-subject abdominal CT registration and unsupervised intra-subject cardiac MR (ACDC dataset) registration, and verified its generalizability in boosting the performance of various pre-trained registration models including registration foundation models and task-specific specialized models.

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

Reg-TTR is a novel efficient test-time refinement framework for medical image registration, which can be incorporated with various pre-trained registration models like registration foundation models (uniGradICON), task-specific specialized models (VoxelMorph, TransMorph) and RDP, MemWarp to achieve superior registration accuracy while maintaining fast inference speed.


## Datasets
Pretrained Weights for uniGradICON 
The pre-trained weights of the uniGradICON registration foundation model used in this project are obtained directly from the official release of the uniGradICON model.  You can find the model weights [here](https://github.com/XiangChen1994/EOIR).

The datasets used are **Abdomen CT** and **ACDC**.

Run the following commands in the `./src` folder to reproduce the results:

```plain
python testabdomen.py -m UniGradICON -d abdomenreg -bs 1 --num_classes 14
python testACDC.py -m UniGradICON -d acdcreg -bs 1 --num_classes 4
```

- `-m UniGradICON`: Model name, set to "UniGradICON".
- `-d abdomenreg`: Dataset used, specifically "abdomenreg".
- `-bs 1`: Batch size, defined as 1.

## Usage
Run the script with the following command in folder `./src` to reproduce the results:
```
python train_registration.py -m EOIR -d abdomenreg -bs 1 --num_classes 14 start_channel=32 --gpu_id 0
python train_registration.py -m EOIR_OASIS -d oasisreg -bs 1 --num_classes 36 start_channel=32 --gpu_id 0
python train_registration_ACDC.py -m EOIR_ACDC -d acdcreg -bs 1 --num_classes 4 start_channel=32 --gpu_id 0
python train_registration_LUMIR.py --model EOIR -d lumirreg -bs 1 start_channel=32 --gpu_id 0 
```

- `-d abdomenreg`: Dataset used, specifically 'abdomenreg'.
- `-m EOIR`: Model name, set to 'EOIR'.
- `-bs 1`: Batch size, defined as 1.
- `start_channel=32`: Number of starting channels (`N_s`), set to 32.


To test the trained model, run the script with the following command in folder `./src` to get the npz files:
```
python test_registration_abdomen.py -m EOIR -d abdomenreg -bs 1 start_channel=32 --gpu_id 0
python test_registration_OASIS.py -m EOIR_OASIS -d oasisreg -bs 1 start_channel=32 --gpu_id 0
python test_registration_ACDC.py -m EOIR_ACDC -d acdcreg -bs 1 start_channel=32 --gpu_id 0 
python test_registration_LUMIR.py -m EOIR -d lumirreg -bs 1 start_channel=32 --gpu_id 0 
```
- `-d abdomenreg`: Dataset used, specifically 'abdomenreg'.
- `-m EOIR`: Model name, set to 'EOIR'.
- `-bs 1`: Batch size, defined as 1.
- `start_channel=32`: Number of starting channels (`N_s`), set to 32.

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
