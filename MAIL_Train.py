from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from Dataset import MARTrainDataset, MARTestDataset
from math import ceil
from utils.evaluate import PSNR
from utils.evaluate import SSIM
from Unet.ConvNet import IterWave

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/train/", help='txt path to training data')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--patchSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=50000, help='total number of training epochs')
parser.add_argument('--h', type=int, default=4, help='the number of feature maps')
parser.add_argument('--T', type=int, default=10, help='Stage number T')
parser.add_argument('--eta1', type=float, default=0.01, help='stepsize for updating M')
parser.add_argument('--eta2', type=float, default=0.01, help='stepsize for updating X')
parser.add_argument('--eta3', type=float, default=0.01, help='stepsize for updating X')
parser.add_argument('--alpha', type=float, default=0.1, help='stepsize for updating X')
parser.add_argument('--beta', type=float, default=0.1, help='stepsize for updating X')
parser.add_argument('--resume', type=int, default=0, help='continue to train from epoch')
parser.add_argument("--milestone", type=int, default=[50,100,150,200], nargs = '+',
                    help="When to decay learning rate")
parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument('--log_dir', default='logs/', help='tensorboard logs')
parser.add_argument('--model_dir', default='models/', help='saving model')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

_modes = ['train', 'val']
# create path
try:
    os.makedirs(opt.log_dir)
except OSError:
    pass
try:
    os.makedirs(opt.model_dir)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

def train_model(net, optimizer, lr_scheduler, datasets):
    batch_size = {'train': opt.batchSize, 'val': 1}
    data_loader = {phase: DataLoader(datasets[phase], batch_size=batch_size[phase], shuffle=True,
                                     num_workers=int(opt.workers)) for phase in _modes}
    num_data = {phase: len(datasets[phase]) for phase in _modes}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in _modes}
    step = 0
    loss_fn = nn.MSELoss()
    for epoch in range(opt.resume, opt.niter):
        mse_per_epoch = {x: 0 for x in _modes}
        tic = time.time()
        # train stage
        lr = optimizer.param_groups[0]['lr']
        phase = 'train'
        optimizer.zero_grad()
        rmse_per_epoch = 0
        for ii, data in enumerate(data_loader[phase]):
            Xma, GT, XLI, mask = [x.cuda() for x in data]
            B, H, W = Xma.shape
            Xma = Xma.view(B, 1, H, W)
            XLI = XLI.view(B, 1, H, W)
            GT = GT.view(B, 1, H, W)
            mask = mask.view(B, 1, H, W)
            Xma = Xma * mask
            GT = GT * mask
            XLI = XLI * mask
            net.train()
            optimizer.zero_grad()
            Yout = net(Xma, XLI)
            loss = loss_fn(Yout, GT)
            loss.backward()
            optimizer.step()

            mse_iter = loss.item()
            mse_per_epoch[phase] += mse_iter
            rmse = torch.sqrt(loss_fn(Yout, GT)).item()
            rmse_per_epoch += rmse
            if ii % 1000 == 0:
                log_str = 'rmse={:5.4f}'
                print(log_str.format(rmse))
            step += 1
        mse_per_epoch[phase] /= (ii + 1)
        rmse_per_epoch /= (ii + 1)
        print('{:s}: Loss={:+.2e}'.format(phase, mse_per_epoch[phase]))
        print('ave_train_rmse={:5.4f}'.format(rmse_per_epoch))
        print('-' * 100)
        # evaluation stage
        net.eval()
        psnr_per_epoch = 0
        ssim_per_epoch = 0
        rmse_per_epoch = 0
        phase = 'val'
        for ii, data in enumerate(data_loader[phase]):
            Xma, GT, XLI, mask = [x.cuda() for x in data]
            B, H, W = Xma.shape
            Xma = Xma.view(B, 1, H, W)
            XLI = XLI.view(B, 1, H, W)
            GT = GT.view(B, 1, H, W)
            mask = mask.view(B, 1, H, W)
            Xma = Xma * mask
            GT = GT * mask
            XLI = XLI * mask
            with torch.set_grad_enabled(False):
                Yout = net(Xma, XLI)
            psnr_iter = PSNR(Yout * mask, GT * mask).item()
            ssim_iter = SSIM(Yout * mask, GT * mask).item()
            rmse = torch.sqrt(loss_fn(Yout, GT)).item()
            psnr_per_epoch += psnr_iter
            ssim_per_epoch += ssim_iter
            rmse_per_epoch += rmse
            if ii % 200 == 0:
                log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>3d}/{:0>3d}, mae={:.2e}'
                print(log_str.format(epoch + 1, opt.niter, phase, ii + 1, num_iter_epoch[phase], rmse))
        psnr_per_epoch /= (ii + 1)
        ssim_per_epoch /= (ii + 1)
        rmse_per_epoch /= (ii + 1)
        print('ave_test_rmse={:4.2f}, ave_test_psnr={:5.4f}, ave_test_ssim={:5.4f}'.format(rmse_per_epoch, psnr_per_epoch, ssim_per_epoch))
        print('-' * 100)
        lr_scheduler.step()
        # save model
        torch.save(net.state_dict(), os.path.join(opt.model_dir, 'MAIL_latest.pt'))
        if (epoch+1) % 1 == 0:
            # save model
            model_prefix = 'model_'
            save_path_model = os.path.join(opt.model_dir, model_prefix + str(epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'step': step + 1,
            }, save_path_model)
            torch.save(net.state_dict(), os.path.join(opt.model_dir, 'MAIL_state_%d.pt' % (epoch+1)))
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc - tic))
    print('Reach the maximal epochs! Finish training')

if __name__ == '__main__':
    def print_network(net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('Total number of parameters: %d' % num_params)
    netWave = IterWave(opt, ch_in=2).cuda()
    print_network(netWave)
    optimizerMAIL = optim.Adam(netWave.parameters(), betas=(0.5, 0.999), lr=opt.lr)
    schedulerMAIL = optim.lr_scheduler.MultiStepLR(optimizerMAIL, milestones=opt.milestone, gamma=0.5)
    # scaler = amp.GradScaler()
    # from opt.resume continue to train
    for _ in range(opt.resume):
        schedulerMAIL.step()
    if opt.resume:
        checkpoint = torch.load(os.path.join(opt.model_dir, 'model_' + str(opt.resume)))
        netWave.load_state_dict(torch.load(os.path.join(opt.model_dir, 'MAIL_state_' + str(opt.resume) + '.pt')))
        print('loaded checkpoints, epoch{:d}'.format(checkpoint['epoch']))
    # load dataset
    train_dataset = MARTrainDataset(opt.data_path, opt.patchSize)
    val_dataset = MARTestDataset("./test_data")
    datasets = {'train': train_dataset, 'val': val_dataset}
    # train model
    train_model(netWave, optimizerMAIL, schedulerMAIL, datasets)
