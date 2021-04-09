import os
import torch
from visdom import Visdom
from util import loadYaml, parseArgs
from network.UNet import ScaleUNet
from dataset.gopro import GoProDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import dataset.util as util


args = parseArgs()
config, saveName = loadYaml(args.config)

os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_available']
device_ids = range(config['gpu_num'])

def print_model(model):
    for n, p in model.named_parameters():
        print(n, p.shape)

def trainScaledUNet():
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
    su, sd = config['scaleup'],config['scaledown']
    scale_ration = 13/16 
    model = ScaleUNet(3, 3, su, sd)
    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)
    print_model(model)

    if config['pretrained_model'] != 'None':
        print('loading Pretrained {}'.format(config['pretrained_model']))
        model.load_state_dict(torch.load(config['pretrained_model']))

    startEpoch = config['start_epoch']
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['step'], gamma=0.5)  # learning rates

    mse = torch.nn.L1Loss()
    viz = Visdom(env=saveName)
    bestPSNR = config['bestPSNR']

    for epoch in range(startEpoch, 1000000):
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
        print("epoch {}: total loss : {:<7.5f}, lr : {}".format(
            epoch, avg_loss, scheduler.get_lr()[0]))
        viz.line(
            X=[epoch],
            Y=[avg_loss],
            win='{}/{}UNettrainMSELoss'.format(su, sd),
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
                print("total PSNR : {:<7.5f}".format(
                    avg_PSNR))
                viz.line(
                    X=[epoch],
                    Y=[avg_PSNR],
                    win='{}/{}UNettestPSNR'.format(su, sd),
                    opts=dict(title='psnr', legend=['valid_psnr']),
                    update='append')
                if avg_PSNR > bestPSNR:
                    bestPSNR = avg_PSNR
                    save_path = os.path.join(config['model_dir'], "{}_{}UNet".format(su,sd) + config['modelName'])
                    if not os.path.exists(config['model_dir']):
                        os.mkdir(config['model_dir'])
                    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    trainScaledUNet()
