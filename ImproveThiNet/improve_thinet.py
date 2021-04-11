import torch
from improve_thinet_utils import improve_thinet_pruned_structure
from network.UNet import UNet
from network.ArbitaryUNet import ArbitaryUNet
from dataset.gopro import GoProDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from util import loadYaml, parseArgs
import os
from visdom import Visdom
import dataset.util as util

args = parseArgs()
config, saveName = loadYaml(args.config)

def train(model, criterion, train_loader, test_loader, fine_tune_epochs, optimizer, scheduler, viz, training_label="",save_dir="./"):
    bestPSNR = 0
    for epoch in range(fine_tune_epochs):
        avg_loss = 0.0
        idx = 0
        model.train()
        for i, train_data in enumerate(train_loader):
            idx += 1
            train_data['L'] = train_data['L'].cuda()
            train_data['H'] = train_data['H'].cuda()
            optimizer.zero_grad()
            sharp = model(train_data['L'])
            loss = criterion(sharp, train_data['H'])
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            if idx % 100 == 0:
                print("{} epoch {}: trained {}".format(training_label,epoch, idx))
        if scheduler is not None:
            scheduler.step()
        avg_loss = avg_loss / idx
        print("{} epoch {}: total loss : {:<7.5f}, lr : {}".format(
            training_label,epoch, avg_loss, scheduler.get_lr()[0]))
        viz.line(
            X=[epoch],
            Y=[avg_loss],
            win='{} trainMSELoss'.format(training_label),
            opts=dict(title='mse loss', legend=['{} train_mse'.format(training_label)]),
            update='append')

        # eval
        if epoch % 10 == 0 or epoch == fine_tune_epochs-1 :
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
                    win='{} testPSNR'.format(training_label),
                    opts=dict(title='psnr', legend=['{} valid_psnr'.format(training_label)]),
                    update='append')
                if avg_PSNR > bestPSNR:
                    bestPSNR = avg_PSNR
                    torch.save(model, os.path.join(save_dir,"{}.pth".format(training_label)))

def print_model(model):
    for n, p in model.named_parameters():
        print(n, p.shape)


def improve_thinet():
    '''
    basic settings and model
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_available']
    # leave 1 gpu for inference when hook
    gpu_id_for_hook = config['gpu_num'] -1
    if config['gpu_num'] == 1:
        device_ids = range(config['gpu_num'])
    else:
        device_ids = range(config['gpu_num']-1)
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
    
    viz = Visdom(env=saveName)
    save_dir = "./pruned_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    criterion = torch.nn.L1Loss()

    model = UNet(3, 3)
    model.load_state_dict(torch.load(config['pretrained_model']))
    print("load weight from path:{}".format(config['pretrained_model']))
    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)
    print_model(model)
    prune_ratio = config['prune_ratio']
    min_channel_ratio = config['min_channel_ratio']
    decay_factor = config['decay_factor']

    '''
    get pruned structure for one-shot
    '''
    pruned_filter_dict, pruned_filter_dict_thinner = improve_thinet_pruned_structure(model, train_loader, r, gpu_id, min_channel_ratio, decay_factor)
    print("pruned_filter_dict:",pruned_filter_dict)
    print("pruned_filter_dict_thinner:",pruned_filter_dict_thinner)

    
    # train from scratch
    # Net1
    training_label = "pruned_structure"
    print("training {}".format(training_label))
    model_path = os.path.join(save_dir, "{}.pth".format(training_label))
    net = ArbitaryUNet(3,3,pruned_filter_dict)
    if os.path.exists(model_path):
        net = torch.load(model_path)

    epoch = 10000
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['step'], gamma=0.5)  # learning rates
    fine_tune(net,criterion, train_loader, test_loader, epoch, optimizer, scheduler, viz, training_label, save_dir)

    # Net2
    training_label = "thinner_pruned_structure"
    print("training {}".format(training_label))
    model_path = os.path.join(save_dir, "{}.pth".format(training_label))
    net = ArbitaryUNet(3,3,pruned_filter_dict_thinner)
    if os.path.exists(model_path):
        net = torch.load(model_path)

    epoch = 10000
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['step'], gamma=0.5)  # learning rates
    fine_tune(net,criterion, train_loader, test_loader, epoch, optimizer, scheduler, viz, training_label, save_dir)

    return

if __name__ == "__main__":
    improve_thinet()
