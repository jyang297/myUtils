import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset


cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name        
        self.h = 256
        self.w = 448
        self.data_root = '/home/jyzhao/Code/Datasets/vimeo_septuplet'
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'sep_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()   
        self.load_data()

    def __len__(self):
        return len(self.meta_data)//3000

    def load_data(self):
        cnt = int(len(self.trainlist) * 0.95)
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]
           
    def crop(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(7):
            imgs[i] = imgs[i][x:x+h, y:y+w, :]
        #img1 = img1[x:x+h, y:y+w, :]
        #gt = gt[x:x+h, y:y+w, :]
        #return img0, gt, img1
        return imgs

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
        # Load images
        # RIFEm with Vimeo-Septuplet
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png', imgpath + '/im4.png', imgpath + '/im5.png', imgpath + '/im6.png', imgpath + '/im7.png']
        # ind = [0, 1, 2, 3, 4, 5, 6]
        imgs = []
        #random.shuffle(ind)
        #ind = ind[:3]
        #ind.sort()
        for i in range(7):
            img = cv2.imread(imgpaths[i])
            imgs.append(img)

        imgs = self.crop(imgs, 224, 224)
        # timestep = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)
        return  imgs

        '''
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        timestep = 0.5
        return img0, gt, img1, timestep
        '''
            
    def __getitem__(self, index):        
        imgs = self.getimg(index)
        if self.dataset_name == 'train':
            #img0, gt, img1 = self.crop(img0, gt, img1, 224, 224)

            # imgs n*[3*H*W]
            if random.uniform(0, 1) < 0.5:
                for i in range(7):
                    imgs[i] = imgs[i][:, :, ::-1]

            if random.uniform(0, 1) < 0.5:
                for i in range(7):
                    imgs[i] = imgs[i][::-1]

            if random.uniform(0, 1) < 0.5:
                for i in range(7):
                    imgs[i] = imgs[i][:, ::-1]
            if random.uniform(0, 1) < 0.5:
                temp = imgs.copy()
                for i in range(7):
                    imgs[i] = temp[6-i]


            # random rotation
            p = random.uniform(0, 1)
            if p < 0.25:
                for i in range(7):
                    imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                for i in range(7):
                    imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_180)
            elif p < 0.75:
                for i in range(7):
                    imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        # img: HWC == permute ==> CHW == concat ==> (C*N)HW  
        tensor_list = [torch.from_numpy(np_img.copy()).permute(2,0,1) for np_img in imgs]
        stacked_imgs = torch.stack(tensor_list, dim=0)
        timestep = torch.tensor(0.5).reshape(1, 1, 1)
        return stacked_imgs, timestep
