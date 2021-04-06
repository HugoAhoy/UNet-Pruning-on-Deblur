import torch
from thinet_utils import get_layers, thinet_prune_layer, save_model, get_conv_nums
from network.UNet import UNet
from dataset.gopro import GoProDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from util import loadYaml, parseArgs
from visdom import Visdom

args = parseArgs()
config, saveName = loadYaml(args.config)

def fine_tune(model, criterion, train_loader, test_loader, epochs, optimizer, scheduler, viz, training_label="",save_dir="./"):
    bestPSNR = 0
    for e in range(epochs):
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
                print("{} fine-tune epoch {}: trained {}".format(training_label,epoch, idx))

        scheduler.step()
        avg_loss = avg_loss / idx
        print("{} fine-tune epoch {}: total loss : {:<4.2f}, lr : {}".format(
            training_label,epoch, avg_loss, scheduler.get_lr()[0]))
        viz.line(
            X=[epoch],
            Y=[avg_loss],
            win='{} trainMSELoss'.format(training_label),
            opts=dict(title='mse loss', legend=['{} train_mse'.format(training_label)]),
            update='append')

        # eval
        if e % 10 == 0 or e == epochs-1 :
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
                    win='{} testPSNR'.format(training_label),
                    opts=dict(title='psnr', legend=['{} valid_psnr'.format(training_label)]),
                    update='append')
                if avg_PSNR > bestPSNR:
                    bestPSNR = avg_PSNR
                    torch.save(model, os.path.join(save_dir,"{}.pth".format(training_label)))

def thinet():
    '''
    basic settings and model
    '''
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
    
    viz = Visdom(env=saveName)
    save_dir = "./pruned_models"
    if not os.path.exists(save_dir):
        os.mkdirs(save_dir)

    criterion = torch.nn.L1Loss()

    model = UNet(3, 3)
    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)
    prune_ratio = 13/16

    # TODO: the setting of optimizer and scheduler
    optimizer = None
    scheduler = None

    '''
    prune layer by layer
    '''
    conv_num = get_conv_nums(model)
    for idx in range(conv_num):
        training_label = "layer_{}".format(idx)
        model_path = os.path.join(save_dir, "{}.pth".format(training_label))
        
        '''
        load the saved model until it is the last one.
        '''
        if os.path.exists(model_path):
            model = torch.load(model_path)
            continue

        model = thinet_prune_layer(model, idx, train_loader, prune_ratio)

        '''
        fine-tune after every layer pruning
        '''
        finetune_epoch = 10
        fine_tune(model, train_loader, test_loader, finetune_epoch, optimizer, scheduler, viz, training_label, save_dir)
    
    # final finetune
    training_label = "final"
    model_path = os.path.join(save_dir, "{}.pth".format(training_label))
    if os.path.exists(model_path):
        model = torch.load(model_path)

    epoch = 200
    fine_tune(model, train_loader, test_loader, epoch, optimizer, scheduler, viz, training_label, save_dir)

    return

if __name__ == "__main__":
    thinet()
