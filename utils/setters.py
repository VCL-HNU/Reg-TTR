import os
import torch
import random
import re
import numpy as np

def setGPU(opt):

    os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu_id']
    if not torch.cuda.is_available():
        raise Exception("No GPU found")
    print("----->>>> GPU %s is set up ..." % opt['gpu_id'])


def setFoldersLoggers(opt, split_id=None):
    opt['data_path'] = os.path.join(opt['datasets_path'], opt['dataset'])

    # 原始日志路径
    base_log = os.path.join(opt['logs_path'], opt['dataset'])

    # ========== 新增：继续训练的路径处理 ==========
    if 'resume_path' in opt and opt['resume_path'] is not None:
        # 尝试从文件名中提取epoch信息
        basename = os.path.basename(opt['resume_path'])

        # 匹配 best_score_X_net_epoch_Y.pth 格式
        epoch_match = re.search(r'epoch[_-]?(\d+)', basename)

        if epoch_match:
            resume_epoch = epoch_match.group(1)
            # 如果用户没有指定suffix，自动生成
            suffix = opt.get('resume_suffix', f'continued_from_epoch{resume_epoch}')
        else:
            suffix = opt.get('resume_suffix', 'continued_training')

        # 创建继续训练的专属目录
        opt['log'] = os.path.join(base_log, suffix)

        print(f"\n{'=' * 60}")
        print(f"CONTINUED TRAINING MODE")
        print(f"Original logs: {base_log}")
        print(f"New logs will be saved to: {opt['log']}")
        print(f"{'=' * 60}\n")
    else:
        opt['log'] = base_log

    # 创建目录
    if not os.path.exists(opt['log']):
        os.makedirs(opt['log'], exist_ok=True)

    print("----->>>> Log path: %s" % opt['log'])
    print("----->>>> Data set path: %s" % opt['data_path'])

def setSeed(seed=0):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True