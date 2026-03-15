import os
import torch
import random
import numpy as np
import nibabel as nib
import torchio as tio
from torch.utils.data import Dataset
from scipy import ndimage
import scipy
import matplotlib


def normalization(data):
    _range = np.max(data) - np.min(data)
    # print('ori min max', np.max(data), np.min(data))
    return (data - np.min(data)) / _range


def mask2onehot(mask, num_classes=4):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    # _mask = [mask == i for i in range(num_classes)]
    # onehot = np.array(_mask).astype(np.uint8).transpose((1,0,2,3,4))[0]
    # _mask = [normalization(ndimage.distance_transform_edt(mask == i)) for i in range(num_classes)]
    # print('unique',np.unique(mask),mask.shape)
    mask = mask[0]
    _mask0 = normalization(ndimage.distance_transform_edt(mask == 0.))
    _mask1 = normalization(ndimage.distance_transform_edt((mask == 1.)))
    _mask2 = normalization(ndimage.distance_transform_edt(mask == 2.))
    _mask3 = normalization(ndimage.distance_transform_edt(mask == 3.))
    _mask = [_mask0, _mask1, _mask2, _mask3]

    # test = ndimage.distance_transform_edt(_mask[1])
    # edt = np.array(_mask).astype(np.uint8).transpose((1,0,2,3,4))[0]
    edt = np.array(_mask)
    # print('onehot, msk',onehot.shape, mask.shape)
    # print('onehot',onehot.shape,onehot[1,50:70,50:70,8],onehot.max())
    # a = np.random.randint(2, size=(4,128,128,16))
    # edt = ndimage.distance_transform_edt(a)
    # edt = edt/edt.max()
    # print('minmax',np.max(edt), np.min(edt))
    # matplotlib.image.imsave('1.jpg', edt[1,:,:,6])
    # matplotlib.image.imsave('2.jpg', _mask1[:,:,6])
    # print('edt',edt.shape,edt[0,50:70,50:70,8],edt.max())
    # input()
    return edt


class acdcreg_loader(Dataset):  # acdcreg_loader

    train_list = [idx for idx in range(1, 17 + 1)] + [idx for idx in range(21, 37 + 1)] \
                 + [idx for idx in range(41, 57 + 1)] + [idx for idx in range(61, 77 + 1)] \
                 + [idx for idx in range(81, 97 + 1)]

    val_list = [idx for idx in range(18, 21)] + [idx for idx in range(38, 41)] \
               + [idx for idx in range(58, 61)] + [idx for idx in range(78, 81)] \
               + [idx for idx in range(98, 100 + 1)]

    test_list = [idx for idx in range(101, 150 + 1)]

    # train_list = [idx for idx in range(1, 4 + 1)]
    # val_list = [idx for idx in range(5,7)]

    def __init__(self,
                 root_dir='./../../../data/acdcreg/',
                 split='train',  # train, val or test
                 intensity_cap=0.001,
                 enable_random_ed_es_flip=1,
                 ):
        self.root_dir = root_dir
        self.split = split
        self.intensity_cap = intensity_cap
        self.enable_random_ed_es_flip = enable_random_ed_es_flip
        # self.emb_path = "/mnt/disk/hy/dataset/acdcreg/dinov2_image_encoder"
        # self.emb_path = "/mnt/disk/hy/dataset/acdcreg/sam_unflip"
        # self.emb_path = "/mnt/disk/hy/dataset/acdcreg/medsam_image_encoder_0"

        if self.split == 'train':
            self.root_dir = os.path.join(self.root_dir, 'train/')
            self.idxs = self.train_list
            # self.emb_path = "/mnt/disk/hy/dataset/acdcreg/sam_train"
        elif self.split == 'val':
            self.root_dir = os.path.join(self.root_dir, 'train/')
            self.idxs = self.val_list
            # self.emb_path = "/mnt/disk/hy/dataset/acdcreg/sam_train"
        elif self.split == 'test':
            self.root_dir = os.path.join(self.root_dir, 'test/')
            self.idxs = self.test_list
            # self.emb_path = "/mnt/disk/hy/dataset/acdcreg/sam_test"
        else:
            raise ValueError('Invalid split name')

        self.init_augmentation()
        self.init_dataset_in_memory()

    def init_augmentation(self):

        self.random_flip = tio.RandomFlip(axes=(0, 1))
        self.random_gamma = tio.RandomGamma(log_gamma=(-0.3, 0.3))

    def init_dataset_in_memory(self):

        self.data = []
        for sub_idx in self.idxs:
            sub_idx_str = 'patient' + str(sub_idx).zfill(3)
            ed_img_fp = os.path.join(self.root_dir, sub_idx_str + '_ed_img.nii.gz')
            es_img_fp = os.path.join(self.root_dir, sub_idx_str + '_es_img.nii.gz')
            ed_seg_fp = os.path.join(self.root_dir, sub_idx_str + '_ed_seg.nii.gz')
            es_seg_fp = os.path.join(self.root_dir, sub_idx_str + '_es_seg.nii.gz')
            # ed_emb_fp = os.path.join(self.emb_path, sub_idx_str + '_ed_img.npy')
            # es_emb_fp = os.path.join(self.emb_path, sub_idx_str + '_es_img.npy')
            # ed_emb_fp = os.path.join(self.emb_path, 'x_sam_' +str(sub_idx)+'.npy')
            # es_emb_fp = os.path.join(self.emb_path, 'y_sam_' +str(sub_idx)+'.npy')
            # ed_autoseg_fp = os.path.join(self.root_dir, sub_idx_str+'_ed_auto0seg.nii.gz')
            # es_autoseg_fp = os.path.join(self.root_dir, sub_idx_str+'_es_autoseg.nii.gz')

            ed_img = nib.load(ed_img_fp).get_fdata()
            es_img = nib.load(es_img_fp).get_fdata()
            ed_seg = nib.load(ed_seg_fp).get_fdata()
            es_seg = nib.load(es_seg_fp).get_fdata()
            # ed_emb = np.array(np.load(ed_emb_fp), dtype='float32')
            # es_emb = np.array(np.load(es_emb_fp), dtype='float32')

            # ed_autoseg = nib.load(ed_autoseg_fp).get_data()
            # es_autoseg = nib.load(es_autoseg_fp).get_data()

            sub = {
                'sub_idx': sub_idx,
                'ed_img': ed_img,
                'es_img': es_img,
                'ed_seg': ed_seg,
                'es_seg': es_seg,
                # 'ed_emb': ed_emb,
                # 'es_emb': es_emb,
                # 'ed_autoseg': ed_autoseg,
                # 'es_autoseg': es_autoseg,
            }
            self.data.append(sub)
            # print(sub_idx)
        print('Number of subjects: ', len(self.data))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):

        sub = self.data[idx]
        ed_img, es_img, ed_seg, es_seg = sub['ed_img'], sub['es_img'], sub['ed_seg'], sub['es_seg']
        # ed_emb = sub['ed_emb']
        # es_emb = sub['es_emb']
        # x, x_seg, x_emb = ed_img, ed_seg, ed_emb
        # y, y_seg, y_emb = es_img, es_seg, es_emb
        x, x_seg = ed_img, ed_seg
        y, y_seg = es_img, es_seg
        # ed_img, es_img, ed_seg, es_seg, ed_autoseg, es_autoseg = sub['ed_img'], sub['es_img'], sub['ed_seg'], sub['es_seg'], sub['ed_autoseg'], sub['es_autoseg']
        # x, x_seg, x_autoseg = ed_img, ed_seg, ed_autoseg
        # y, y_seg, y_autoseg = es_img, es_seg, es_autoseg

        if self.enable_random_ed_es_flip and random.random() > 0.5:
            x, y = es_img, ed_img
            x_seg, y_seg = es_seg, ed_seg
            # x_emb, y_emb = es_emb, ed_emb
            # x_autoseg, y_autoseg = es_autoseg, ed_autoseg

        x, x_seg = np.ascontiguousarray(x), np.ascontiguousarray(x_seg)
        y, y_seg = np.ascontiguousarray(y), np.ascontiguousarray(y_seg)
        # x, x_seg, x_autoseg = np.ascontiguousarray(x), np.ascontiguousarray(x_seg), np.ascontiguousarray(x_autoseg)
        # y, y_seg, y_autoseg = np.ascontiguousarray(y), np.ascontiguousarray(y_seg), np.ascontiguousarray(y_autoseg)

        # 0-1
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))

        x, x_seg = x[None, ...], x_seg[None, ...]
        y, y_seg = y[None, ...], y_seg[None, ...]
        # x_emb, y_emb = x_emb[0], y_emb[0]

        # x_seg_mask, y_seg_mask = mask2onehot(x_seg), mask2onehot(y_seg)
        #########
        x_seg_mask, y_seg_mask = x_seg, y_seg

        #########

        x, x_seg, x_seg_mask = torch.from_numpy(x), torch.from_numpy(x_seg), torch.from_numpy(x_seg_mask)
        y, y_seg, y_seg_mask = torch.from_numpy(y), torch.from_numpy(y_seg), torch.from_numpy(y_seg_mask)
        # x, x_seg, x_seg_mask, x_emb = torch.from_numpy(x), torch.from_numpy(x_seg), torch.from_numpy(x_seg_mask), torch.from_numpy(x_emb)
        # y, y_seg, y_seg_mask, y_emb = torch.from_numpy(y), torch.from_numpy(y_seg), torch.from_numpy(y_seg_mask), torch.from_numpy(y_emb)

        if self.split == 'train' and self.enable_random_ed_es_flip:
            subject = tio.Subject(
                img1=tio.ScalarImage(tensor=x),
                img2=tio.ScalarImage(tensor=y),
                msk1=tio.LabelMap(tensor=x_seg),
                msk2=tio.LabelMap(tensor=y_seg),
                edt_msk1=tio.LabelMap(tensor=x_seg_mask),
                edt_msk2=tio.LabelMap(tensor=y_seg_mask),
                # emb1 = tio.LabelMap(tensor=x_emb),
                # emb2 = tio.LabelMap(tensor=y_emb),
            )
            subject = self.random_flip(subject)
            x, x_seg, x_seg_mask = subject.img1.data, subject.msk1.data, subject.edt_msk1.data
            y, y_seg, y_seg_mask = subject.img2.data, subject.msk2.data, subject.edt_msk2.data
            # x, x_seg, x_seg_mask, x_emb = subject.img1.data, subject.msk1.data, subject.edt_msk1.data, subject.emb1.data
            # y, y_seg, y_seg_mask, y_emb = subject.img2.data, subject.msk2.data, subject.edt_msk2.data, subject.emb2.data

        return x, x_seg, y, y_seg, x_seg_mask, y_seg_mask, sub['sub_idx']

        # return x, x_seg, y, y_seg, x_seg_mask, y_seg_mask, x_emb, y_emb, sub['sub_idx']
