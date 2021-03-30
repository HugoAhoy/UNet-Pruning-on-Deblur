import torch
from thinet_utils import get_layers, thinet_prune_layer, save_model
from network.UNet import UNet
from dataset.gopro import GoProDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from util import loadYaml, parseArgs

args = parseArgs()
config, saveName = loadYaml(args.config)

def fine_tune(model, trainSet, testSet, epochs, optimizer, scheduler)
    for e in epoch:
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
        # viz.line(
        #     X=[epoch],
        #     Y=[avg_loss],
        #     win='trainMSELoss',
        #     opts=dict(title='mse', legend=['train_mse']),
        #     update='append')

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

    model = UNet(3, 3)
    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)

    # TODO: the setting of optimizer and scheduler
    optimizer = None
    scheduler = None

    '''
    prune layer by layer
    '''
    layers = get_layers(model)
    for layer in layers:
        model = thinet_prune_layer(model, layer)

        '''
        fine-tune after every layer pruning
        '''
        epoch = 2 # as mentioned in paper, fine-tune 1 or 2 epochs every pruning
        fine_tune(model, trainSet, testSet, epoch, optimizer, scheduler)
    
    epoch = 100
    fine_tune(model, trainSet, testSet, epoch, optimizer, scheduler)

    save_model()
    return

if __name__ == "__main__":
    thinet()
