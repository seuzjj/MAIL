import os
import os.path
import numpy as np
import random
import h5py
import torch
import torch.utils.data as udata
from numpy.random import RandomState


def image_get_minmax():
    return 0.0, 1.0


def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data * 2.0 - 1.0
    data = data.astype(np.float32)
    data = np.transpose(np.expand_dims(data, 2), (2, 0, 1))
    return data

# image augment
def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

class MARTrainDataset(udata.Dataset):
    def __init__(self, dir, patchSize):
        super().__init__()
        self.dir = dir
        self.patch_size = patchSize
        self.sample_num = 1
        self.txtdir = os.path.join(self.dir, 'data.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.rand_state = RandomState(66)
        self.each_image_num = 10

    def __len__(self):
        return self.sample_num * self.each_image_num

    def __getitem__(self, idx):
        data_file = self.mat_files[(idx//self.each_image_num)][:-1]
        abs_dir = os.path.join(self.dir, data_file)
        m_dir = os.path.join(self.dir, 'mask/mask.h5')
        mask_file = h5py.File(m_dir, 'r')
        M = mask_file['mask'][()]
        mask_file.close()
        file = h5py.File(abs_dir, 'r')
        Xma = file['ma_CT'][()]
        XLI = file['LI_CT'][()]
        Xgt = file['gt_CT'][()]
        file.close()
        remain = idx % self.each_image_num
        M = M[remain,:,:]
        Xma = Xma[remain,:,:]
        XLI = XLI[remain,:,:]
        Xgtclip = np.clip(Xgt,0,1)
        Xgtnorm = Xgtclip
        Xmaclip = np.clip(Xma, 0, 1)
        Xmanorm = Xmaclip
        XLIclip = np.clip(XLI, 0,1)
        XLInorm = XLIclip
        O = Xmanorm*255
        B = Xgtnorm*255
        LI = XLInorm*255
        O = O.astype(np.float32)
        LI = LI.astype(np.float32)
        B = B.astype(np.float32)
        Mask = M.astype(np.float32)
        non_Mask = 1 - Mask #non_metal region

        #image augment(train only)
        mode = random.randint(0, 7)
        O = augment_img(O, mode=mode)
        B = augment_img(B, mode=mode)
        LI = augment_img(LI, mode=mode)
        non_Mask = augment_img(non_Mask, mode=mode)
        return torch.from_numpy(O.copy()),torch.from_numpy(B.copy()),torch.from_numpy(LI.copy()),torch.from_numpy(non_Mask.copy())


class MARTestDataset(udata.Dataset):
    def __init__(self, dir):
        super().__init__()
        self.dir = dir
        self.txtdir = os.path.join(self.dir, 'data.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.sample_num = 1
        self.each_image_num = 10

    def __len__(self):
        return int(self.sample_num) * self.each_image_num

    def __getitem__(self, idx):
        data_file = self.mat_files[idx // self.each_image_num][:-1]
        abs_dir = os.path.join(self.dir, data_file)
        m_dir = os.path.join(self.dir, 'mask/mask.h5')
        mask_file = h5py.File(m_dir, 'r')
        M = mask_file['mask'][()]
        mask_file.close()
        file = h5py.File(abs_dir, 'r')
        Xma = file['ma_CT'][()]
        XLI = file['LI_CT'][()]
        Xgt = file[('gt_CT')][()]
        file.close()
        remain = idx % self.each_image_num
        M = M[remain, :, :]
        Xma = Xma[remain, :, :]
        XLI = XLI[remain, :, :]
        Xgtclip = np.clip(Xgt, 0, 1)
        Xgtnorm = Xgtclip
        Xmaclip = np.clip(Xma, 0, 1)
        Xmanorm = Xmaclip
        XLIclip = np.clip(XLI, 0, 1)
        XLInorm = XLIclip
        O = Xmanorm * 255
        B = Xgtnorm * 255
        LI = XLInorm * 255
        O = O.astype(np.float32)
        LI = LI.astype(np.float32)
        B = B.astype(np.float32)
        Mask = M.astype(np.float32)
        non_Mask = 1 - Mask
        return torch.from_numpy(O.copy()), torch.from_numpy(B.copy()), torch.from_numpy(LI.copy()), torch.from_numpy(
            non_Mask.copy())
