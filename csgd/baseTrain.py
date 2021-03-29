import os
import torch
from visdom import Visdom
from network.UNet import UNet
from dataset.gopro import GoProDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import dataset.util as util
from collections import OrderedDict

def save_pth(pthdict, path):
    save_dict = {}
    new_state_dict = OrderedDict()
    num_params = 0
    for k, v in pthdict['model'].items():
        key = k
        if k.split('.')[0] == 'module':
            key = k[7:]
        new_state_dict[key] = v
        num_params += v.numel()
    save_dict['model'] = new_state_dict

    if 'deps' in pthdict:
        save_dict['deps'] = pthdict['deps']
    torch.save(save_dict, path)
    print('save {} values to {} model'.format(len(save_dict['model']), path))

def trainUNet(config, savePath):
    saveName = "%s-%s.yaml" % (config['note'], config['dataset'])
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_available']
    device_ids = range(config['gpu_num'])
    trainSet = GoProDataset(sharp_root=config['train_sharp'], blur_root=config['train_blur'],
                            resize_size=config['resize_size'], patch_size=config['crop_size'],
                            phase='train')
    testSet = GoProDataset(sharp_root=config['test_sharp'], blur_root=config['test_blur'],
                           resize_size=config['resize_size'], patch_size=config['crop_size'],
                           phase='test')

    train_loader = DataLoader(trainSet,
                              batch_size=config['batchsize'],
                              shuffle=True, num_workers=4,
                              drop_last=True, pin_memory=True)
    test_loader = DataLoader(testSet, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

    model = UNet(3, 3)
    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)

    if config['pretrained_model'] != 'None':
        print('loading Pretrained {}'.format(config['pretrained_model']))
        model.load_state_dict(torch.load(config['pretrained_model']))

    startEpoch = config['start_epoch']
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['step'], gamma=0.5)  # learning rates

    mse = torch.nn.L1Loss()
    viz = Visdom(env=saveName)
    bestPSNR = config['bestPSNR']
    baseTrainMaxEpoch = config['base_train_max_epochs']

    for epoch in range(startEpoch, baseTrainMaxEpoch):
        avg_loss = 0.0
        idx = 0
        model.train()
        for i, train_data in enumerate(train_loader):
            idx += 1
            train_data['L'] = train_data['L'].cuda()
            train_data['H'] = train_data['H'].cuda()
            optimizer.zero_grad()
            sharp = model(train_data['L'])
            loss = mse(sharp, train_data['H'])
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            if idx % 100 == 0:
                print("epoch {}: trained {}".format(epoch, idx))

        scheduler.step()
        avg_loss = avg_loss / idx
        print("epoch {}: total loss : {:<4.2f}, lr : {}".format(
            epoch, avg_loss, scheduler.get_lr()[0]))
        viz.line(
            X=[epoch],
            Y=[avg_loss],
            win='trainMSELoss',
            opts=dict(title='mse', legend=['train_mse']),
            update='append')

        if epoch % config['save_epoch'] == 0:
            with torch.no_grad():
                model.eval()
                avg_PSNR = 0
                idx = 0
                for test_data in test_loader:
                    idx += 1
                    test_data['L'] = test_data['L'].cuda()
                    sharp = model(test_data['L'])
                    sharp = sharp.detach().float().cpu()
                    sharp = util.tensor2uint(sharp)
                    test_data['H'] = util.tensor2uint(test_data['H'])
                    current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)

                    avg_PSNR += current_psnr
                    if idx % 100 == 0:
                        print("epoch {}: tested {}".format(epoch, idx))
                avg_PSNR = avg_PSNR / idx
                print("total PSNR : {:<4.2f}".format(
                    avg_PSNR))
                viz.line(
                    X=[epoch],
                    Y=[avg_PSNR],
                    win='testPSNR',
                    opts=dict(title='psnr', legend=['valid_psnr']),
                    update='append')
                if avg_PSNR > bestPSNR:
                    bestPSNR = avg_PSNR
                    state_dict = model.state_dict()
                    for key, param in state_dict.items():
                        state_dict[key] = param.cpu()
                    save_pth({'model':state_dict}, savePath)
