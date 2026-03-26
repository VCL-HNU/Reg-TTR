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

    base_log = os.path.join(opt['logs_path'], opt['dataset'])

    if 'resume_path' in opt and opt['resume_path'] is not None:
        basename = os.path.basename(opt['resume_path'])

        epoch_match = re.search(r'epoch[_-]?(\d+)', basename)

        if epoch_match:
            resume_epoch = epoch_match.group(1)
            suffix = opt.get('resume_suffix', f'continued_from_epoch{resume_epoch}')
        else:
            suffix = opt.get('resume_suffix', 'continued_training')

        opt['log'] = os.path.join(base_log, suffix)

        print(f"\n{'=' * 60}")
        print(f"CONTINUED TRAINING MODE")
        print(f"Original logs: {base_log}")
        print(f"New logs will be saved to: {opt['log']}")
        print(f"{'=' * 60}\n")
    else:
        opt['log'] = base_log

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
