import random
import numpy as np
import torch
import cv2
import torch.utils.data as data
import dataset.util as util


class GoProDataset(data.Dataset):

    def __init__(self, sharp_root, blur_root, resize_size, patch_size, phase='train'):
        super(GoProDataset).__init__()

        self.n_channels = 3
        self.resize_size = resize_size
        self.patch_size = patch_size
        self.phase = phase
        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(sharp_root)
        self.paths_L = util.get_image_paths(blur_root)

        assert self.paths_H, 'Error: Sharp path is empty.'
        assert self.paths_L, 'Error: Blur path is empty.'

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        L_path = self.paths_L[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_L = util.imread_uint(L_path, self.n_channels)

        # ------------------------------------
        # sythesize L image via matlab's bicubic
        # ------------------------------------
        H, W, _ = img_H.shape

        if self.phase == 'train':
            if self.resize_size != 0:
                img_L = cv2.resize(img_L, (self.resize_size, self.resize_size))
                img_H = cv2.resize(img_H, (self.resize_size, self.resize_size))
            if self.patch_size != 0:
                H, W, C = img_L.shape

                # --------------------------------
                # randomly crop L patch
                # --------------------------------
                rnd_h = random.randint(0, max(0, H - self.patch_size))
                rnd_w = random.randint(0, max(0, W - self.patch_size))
                img_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
                img_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = np.random.randint(0, 8)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

            # --------------------------------
            # get patch pairs
            # --------------------------------
            img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)
            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        else:
            img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)
            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time

    TRAIN_SHARP = '/dataset/REDS_AUG/train/sharp/'
    TRAIN_BLUR = '/dataset/REDS_AUG/train/sharp/'
    trainSet = GoProDataset(sharp_root=TRAIN_SHARP, blur_root=TRAIN_SHARP, patch_size=100, resize_size=0, phase='train')

    train_loader = DataLoader(trainSet,
                              batch_size=64,
                              shuffle=True,
                              num_workers=1,
                              drop_last=True,
                              pin_memory=True)
    avg_time = 0
    start = time.time()
    idx = 0
    for i, train_data in enumerate(train_loader):
        end = time.time()
        avg_time += end - start
        print(end - start)
        start = end
        idx += 1

    print(avg_time / idx)
