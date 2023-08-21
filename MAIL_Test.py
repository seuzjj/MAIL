from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
from utils.evaluate import PSNR
from utils.evaluate import SSIM
from Unet.ConvNet import IterWave
import h5py

parser = argparse.ArgumentParser(description="MAILNet_Test")
parser.add_argument("--model_dir", type=str, default="models/MAILNet_latest.pt", help='path to model file')
parser.add_argument("--data_path", type=str, default="./test_data/", help='path to test data')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--save_path", type=str, default="save_results/", help='path to testing results')
parser.add_argument('--T', type=int, default=4, help='Stage number T')
parser.add_argument('--h', type=int, default=4, help='the number of feature maps')
parser.add_argument('--eta1', type=float, default=0.01, help='stepsize for updating M')
parser.add_argument('--eta2', type=float, default=0.01, help='stepsize for updating X')
parser.add_argument('--eta3', type=float, default=0.01, help='stepsize for updating X')
parser.add_argument('--alpha', type=float, default=0.1, help='stepsize for updating X')
parser.add_argument('--beta', type=float, default=0.1, help='stepsize for updating X')
parser.add_argument('--batchSize', type=int, default=1, help='testing input batch size')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def test_image(data_path):
    abs_dir = data_path + '/0001.h5'
    m_dir = data_path + "/mask/mask.h5"
    mask_file = h5py.File(m_dir, 'r')
    M = mask_file['mask'][()]
    mask_file.close()
    file = h5py.File(abs_dir, 'r')
    Xma = file['ma_CT'][()]
    XLI = file['LI_CT'][()]
    Xgt = file[('gt_CT')][()]
    file.close()
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
    return torch.from_numpy(O.copy()), torch.from_numpy(LI.copy()), torch.from_numpy(B.copy()), torch.from_numpy(
        non_Mask.copy())


def main():
    # Build model
    print('Loading model ...\n')
    model = IterWave(opt, ch_in=2).cuda()
    model.load_state_dict(torch.load(opt.model_dir))
    model.eval()

    data_path = opt.data_path
    loss_fn = nn.MSELoss()
    ave_psnr = 0
    ave_ssim = 0
    ave_rmse = 0

    ma, LI, gt, M= test_image(data_path)
    B, H, W = ma.shape
    gt = gt.view(1, 1, H, W).cuda()
    for i in range(B):
        Xma = ma[i,:,:].view(1, 1, H, W).cuda()
        XLI = LI[i,:,:].view(1, 1, H, W).cuda()
        mask = M[i,:,:].view(1, 1, H, W).cuda()
        Xma = Xma * mask
        GT = gt * mask
        XLI = XLI * mask
        Yout = model(Xma, XLI)
        Y = Yout * mask
        psnr_iter = PSNR(Y, GT)
        ssim_iter = SSIM(Y, GT)
        rmse_iter = torch.sqrt(loss_fn(Y, GT)).item()
        ave_psnr += psnr_iter
        ave_ssim += ssim_iter
        ave_rmse += rmse_iter
    ave_psnr /= B
    ave_ssim /= B
    ave_rmse /= B
    print('ave_test_rmse={:4.2f}, ave_test_psnr={:5.4f}, ave_test_ssim={:5.4f}'.format(ave_rmse, ave_psnr, ave_ssim))


if __name__ == "__main__":
    main()
