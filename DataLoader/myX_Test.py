import numpy as np
from glob import glob
import os
import random
from PIL import ImageEnhance, Image
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 
from torch.utils.tensorboard import SummaryWriter


class X_Test(Dataset):
    def make_2D_dataset_X_Test(test_data_path   ):
        """ make [I0,I1,It,t,scene_folder] """
        """ 1D (accumulated) """
        testPath = []
        t_step_size=7
        for type_folder in sorted(glob(os.path.join(test_data_path, '*', ''))):  # [type1,type2,type3,...]
            for scene_folder in sorted(glob(os.path.join(type_folder, '*', ''))):  # [scene1,scene2,..]
                frame_folder = sorted(glob(pathname=scene_folder + '*.png'))  # 32 multiple, ['00000.png',...,'00032.png']
                for idx in range(0, len(frame_folder), t_step_size):  # 0,32,64,...
                    if idx >= len(frame_folder) - 7:
                        break

                    Image0to6paths = []
                    for mul in range(7):
                        Image0to6paths.append(frame_folder[idx + mul])  # It

                    testPath.append(Image0to6paths)
        return testPath


    def frames_loader_test(Image0to6Path):
        frames = []
        for path in Image0to6Path:
            frame = cv2.imread(path)
            frames.append(frame)
        (ih, iw, c) = frame.shape
        frames = np.stack(frames, axis=0)  # (T, H, W, 3)

        """ np2Tensor [-1,1] normalized """
        frames = X_Test.RGBframes_np2Tensor(frames)
        stacked_frames = frames.view(-1, frames.shape[2], frames.shape[3])
        return stacked_frames


    def RGBframes_np2Tensor(imgIn, channel=3):
        ## input : T, H, W, C
        if channel == 1:
            # rgb --> Y (gray)
            imgIn = np.sum(
                    imgIn * np.reshape(
                        [65.481, 128.553, 24.966], [1, 1, 1, 3]
                        ) / 255.0,
                    axis=3,
                    keepdims=True) + 16.0

        # to Tensor
        ts = (0, 3, 1, 2)  ############# dimension order should be [T, C, H, W]
        imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)

        return imgIn

    def __init__(self, test_data_path, multiple):
        self.test_data_path = test_data_path
        self.multiple = multiple
        self.testPath = X_Test.make_2D_dataset_X_Test(
                self.test_data_path, multiple, t_step_size=32)

        self.nIterations = len(self.testPath)

        # Raise error if no images found in test_data_path.
        if len(self.testPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " \
                    + self.test_data_path + "\n"))


    def __getitem__(self, idx):
        Image0to6Path_selected = self.testPath[idx]

        
        frames = X_Test.frames_loader_test(Image0to6Path_selected)
        # including "np2Tensor [-1,1] normalized"



        return frames

    def __len__(self):
        return self.nIterations


