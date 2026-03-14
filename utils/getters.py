import re
import os
import glob
import torch
import numpy as np

from torch.utils.data import DataLoader

from models import getModel
from loaders.acdcreg_loader import acdcreg_loader
from loaders.abdomenreg_loader import abdomenreg_loader
from utils.functions import modelSaver, convert_state_dict


def loadDataset(opt, split='train'):
    dataset_name = opt['dataset']
    data_path = opt['data_path']

    if dataset_name == 'acdcreg':
        loader = acdcreg_loader(root_dir=data_path, split=split)
    elif dataset_name == 'abdomenreg':
        loader = abdomenreg_loader(root_dir=data_path, split=split)
    else:
        raise ValueError('Unkown datasets: please define proper dataset name')

    print("----->>>> %s dataset is loaded ..." % dataset_name)

    return loader


def getDataLoader(opt, split='train'):
    if split == 'train':
        data_shuffle = True
        batch_size = opt['batch_size']
    else:
        data_shuffle = False
        batch_size = 1

    num_workers = opt['num_workers']
    print("----->>>> Loading %s dataset ..." % (split))
    dataset = loadDataset(opt, split)
    loader = DataLoader(dataset=dataset,
                        num_workers=num_workers,
                        batch_size=batch_size,
                        pin_memory=True,
                        shuffle=data_shuffle)
    print("----->>>> %s batch size: %d, # of %s iterations per epoch: %d" % (
    split, batch_size, split, int(len(dataset) / batch_size)))

    return loader


def getModelSaver(opt, suffix=None):
    if suffix is None:
        model_saver = modelSaver(opt['log'], opt['save_freq'], opt['n_checkpoints'])
    else:
        sv_path = os.path.join(opt['log'], suffix)
        os.makedirs(sv_path, exist_ok=True)
        model_saver = modelSaver(sv_path, opt['save_freq'], opt['n_checkpoints'])

    return model_saver


def findLastCheckpoint(save_path):
    file_list = glob.glob(os.path.join(save_path, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall("net_epoch_(.*)_score_.*.pth.*", file_)
            if result:
                epochs_exist.append(int(result[0]))
        init_epoch = max(epochs_exist)
    else:
        init_epoch = 0

    score = None
    if init_epoch > 0:
        for file_ in file_list:
            file_name = "net_epoch_" + str(init_epoch) + "_score_(.*).pth.*"
            result = re.findall(file_name, file_)
            if result:
                score = result[0]
                break

    return_name = None
    if init_epoch > 0:
        return_name = "net_epoch_" + str(init_epoch) + "_score_" + score + ".pth"

    return init_epoch, score, return_name


def findBestCheckpoint(save_path):
    file_list = glob.glob(os.path.join(save_path, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        scores = []
        for file_ in file_list:
            result = re.findall("best_score_(.*)_net_epoch_.*.pth.*", file_)
            if result:
                epochs_exist.append(result[0])
                scores.append(float(result[0]))
        ind = np.argmax(scores)
        score = epochs_exist[ind]
        for file_ in file_list:
            file_name = "best_score_" + str(score) + "_net_epoch_.*.pth.*"
            result = re.findall(file_name, file_)
            if result:
                return_name = result[0]
                file_name = "best_score_" + str(score) + "_net_epoch_(.*).pth.*"
                result = re.findall(file_name, file_)
                epoch = result[0]
                return epoch, score, return_name

    raise ValueError("can't find checkpoints")


def findCheckpointByEpoch(save_path, epoch):
    file_list = glob.glob(os.path.join(save_path, '*epoch*.pth'))
    if file_list:
        for file_ in file_list:
            file_name = "net_epoch_" + str(epoch) + "_score_.*.pth.*"
            result = re.findall(file_name, file_)
            if result:
                return result[0]

    raise ValueError("can't find checkpoints")


def findBestDiceByEpoch(save_path, epoch):
    file_list = glob.glob(os.path.join(save_path, '*epoch*.pth'))
    if file_list:
        for file_ in file_list:
            file_name = "best_score_.*_net_epoch_" + str(epoch) + ".pth.*"
            result = re.findall(file_name, file_)
            if result:
                return result[0]

    raise ValueError("can't find checkpoints")


def getTrainModelWithCheckpoints(opt, model_type=None):
    """
    加载训练模型和检查点
    新增：支持从指定checkpoint继续训练
    """
    print("----->>>> Loading model %s " % opt['model'])

    init_epoch = 0
    model = getModel(opt)
    optimizer_state = None  # 用于返回优化器状态

    # ========== 新增逻辑：resume模式 ==========
    if 'resume_path' in opt and opt['resume_path'] is not None:
        if os.path.exists(opt['resume_path']):
            print(f"\n{'=' * 60}")
            print(f"RESUMING TRAINING FROM CHECKPOINT")
            print(f"Checkpoint: {opt['resume_path']}")
            print(f"{'=' * 60}\n")

            checkpoint = torch.load(opt['resume_path'])

            # 1. 加载模型权重
            if 'state_dict' in checkpoint:
                states = convert_state_dict(checkpoint['state_dict'])
            else:
                # 兼容旧格式（直接是state_dict）
                states = convert_state_dict(checkpoint)
            model.load_state_dict(states)
            print("----->>>> Model weights loaded ✓")

            # 2. 加载epoch信息
            if 'epoch' in checkpoint:
                init_epoch = checkpoint['epoch'] + 1
                print(f"----->>>> Will resume from epoch {init_epoch}")

            # 3. 加载优化器状态
            if 'optimizer' in checkpoint:
                optimizer_state = checkpoint['optimizer']
                print("----->>>> Optimizer state loaded ✓")
            else:
                print("----->>>> Warning: No optimizer state in checkpoint")

            # 4. 显示之前的分数
            if 'score' in checkpoint:
                print(f"----->>>> Previous score: {checkpoint['score']:.4f}")

            return model, init_epoch, optimizer_state
        else:
            print(f"----->>>> ERROR: Resume path not found: {opt['resume_path']}")
            print("----->>>> Training from scratch")
            return model, 0, None

    # ========== 原有逻辑保持不变 ==========
    if model_type is None:
        return model, init_epoch, None

    print("----->>>> Loading model from %s " % opt['log'])
    if model_type == 'last':
        init_epoch, score, file_name = findLastCheckpoint(opt['log'])
    elif model_type == 'best':
        init_epoch, score, file_name = findBestCheckpoint(opt['log'])
    else:
        if 'best' in model_type:
            st = model_type.split('_')[-1]
            opt['log'] = os.path.join(opt['log'], st)
            init_epoch, score, file_name = findBestCheckpoint(opt['log'])

    init_epoch = int(init_epoch)
    if init_epoch > 0:
        print("----->>>> Resuming model by loading epoch %s with dice %s" % (init_epoch, score))
        checkpoint = torch.load(os.path.join(opt['log'], file_name))

        if 'state_dict' in checkpoint:
            states = convert_state_dict(checkpoint['state_dict'])
        else:
            states = convert_state_dict(checkpoint)
        model.load_state_dict(states)

        if 'optimizer' in checkpoint:
            optimizer_state = checkpoint['optimizer']

    return model, init_epoch


def getTestModelWithCheckpoints(opt):
    """
    加载测试模型和权重
    支持直接指定权重文件路径
    """
    model = getModel(opt)

    # UniGradICON 使用预训练权重,跳过检查点加载
    if opt['model'] in ['UniGradICON', 'UniGradICONWrapper', 'UniGradICON_ConvexAdam_Hybrid']:
        print(f"{opt['model']} uses pretrained weights, skipping checkpoint loading")
        info = {
            "file_name": None,
            "epoch": 0,
            "score": 0.0,
        }
        return model, info

    # 如果 load_ckpt 是一个存在的 .pth 文件路径,直接加载
    if opt['load_ckpt'].endswith('.pth') and os.path.exists(opt['load_ckpt']):
        print(f"----->>>> Loading weights from: {opt['load_ckpt']}")
        states = convert_state_dict(torch.load(opt['load_ckpt']))
        model.load_state_dict(states)

        # 尝试从文件名解析 epoch 和 score
        basename = os.path.basename(opt['load_ckpt'])
        epoch = '0'
        score = '0.0'

        # 尝试匹配 best_score_X_net_epoch_Y.pth
        match = re.search(r'best_score_([\d\.]+)_net_epoch_(\d+)', basename)
        if match:
            score = match.group(1)
            epoch = match.group(2)
        else:
            # 尝试匹配 net_epoch_X_score_Y.pth
            match = re.search(r'net_epoch_(\d+)_score_([\d\.]+)', basename)
            if match:
                epoch = match.group(1)
                score = match.group(2)

        print(f"----->>>> Weights loaded successfully [epoch: {epoch}, score: {score}]")

        info = {
            "file_name": opt['load_ckpt'],
            "epoch": int(epoch),
            "score": float(score),
        }
        return model, info

    # 如果是 'none',不加载权重
    if opt['load_ckpt'] == 'none':
        print("----->>>> No weights loaded")
        info = {
            "file_name": None,
            "epoch": 0,
            "score": 0.0,
        }
        return model, info

    # 否则报错
    raise ValueError(
        f"Invalid load_ckpt: {opt['load_ckpt']}\n"
        f"Please provide:\n"
        f"  1. Full path to .pth file (e.g., /path/to/model.pth)\n"
        f"  2. 'none' to skip loading weights"
    )