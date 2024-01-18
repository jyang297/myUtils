from __future__ import division
import os, glob, sys, torch, shutil, random, math, time, cv2
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from datetime import datetime
from torch.nn import init
from skimage.measure import compare_ssim
# from skimage.metrics import structural_similarity
from torch.autograd import Variable
from torchvision import models

class X_Train(data.Dataset):
    def __init__(self, args, max_t_step_size):
        self.args = args
        self.max_t_step_size = max_t_step_size

        self.framesPath = make_2D_dataset_X_Train(self.args.train_data_path)
        self.nScenes = len(self.framesPath)

        # Raise error if no images found in train_data_path.
        if self.nScenes == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.args.train_data_path + "\n"))

    def __getitem__(self, idx):
        t_step_size = random.randint(2, self.max_t_step_size)
        t_list = np.linspace((1 / t_step_size), (1 - (1 / t_step_size)), (t_step_size - 1))

        candidate_frames = self.framesPath[idx]
        firstFrameIdx = random.randint(0, (64 - t_step_size))
        interIdx = random.randint(1, t_step_size - 1)  # relative index, 1~self.t_step_size-1
        interFrameIdx = firstFrameIdx + interIdx  # absolute index
        t_value = t_list[interIdx - 1]  # [0,1]

        if (random.randint(0, 1)):
            frameRange = [firstFrameIdx, firstFrameIdx + t_step_size, interFrameIdx]
        else:  ## temporally reversed order
            frameRange = [firstFrameIdx + t_step_size, firstFrameIdx, interFrameIdx]
            interIdx = t_step_size - interIdx  # (self.t_step_size-1) ~ 1
            t_value = 1.0 - t_value

        frames = frames_loader_train(self.args, candidate_frames,
                                     frameRange)  # including "np2Tensor [-1,1] normalized"

        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0)

    def __len__(self):
        return self.nScenes
    
def make_2D_dataset_X_Train(dir):
    framesPath = []
    # Find and loop over all the clips in root `dir`.
    for scene_path in sorted(glob.glob(os.path.join(dir, '*', ''))):
        sample_paths = sorted(glob.glob(os.path.join(scene_path, '*', '')))
        for sample_path in sample_paths:
            frame65_list = []
            for frame in sorted(glob.glob(os.path.join(sample_path, '*.png'))):
                frame65_list.append(frame)
            framesPath.append(frame65_list)

    print("The number of total training samples : {} which has 65 frames each.".format(
        len(framesPath)))  ## 4408 folders which have 65 frames each
    return framesPath


def frames_loader_train(args, candidate_frames, frameRange):
    frames = []
    for frameIndex in frameRange:
        frame = cv2.imread(candidate_frames[frameIndex])
        frames.append(frame)
    (ih, iw, c) = frame.shape
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)
    if args.need_patch:  ## random crop
        ps = args.patch_size
        ix = random.randrange(0, iw - ps + 1)
        iy = random.randrange(0, ih - ps + 1)
        frames = frames[:, iy:iy + ps, ix:ix + ps, :]

    if random.random() < 0.5:  # random horizontal flip
        frames = frames[:, :, ::-1, :]

    # No vertical flip

    rot = random.randint(0, 3)  # random rotate
    frames = np.rot90(frames, rot, (1, 2))

    """ np2Tensor [-1,1] normalized """
    frames = RGBframes_np2Tensor(frames, args.img_ch)

    return frames


def crop(self, img0, gt, img1, h, w):
    # input  H, W, C
    ih, iw, _ = img0.shape
    x = np.random.randint(0, ih - h + 1)
    y = np.random.randint(0, iw - w + 1)
    img0 = img0[x:x+h, y:y+w, :]
    img1 = img1[x:x+h, y:y+w, :]
    gt = gt[x:x+h, y:y+w, :]
    return img0, gt, img1

def RGBframes_np2Tensor(imgIn, channel=3):
    ## input : T, H, W, C
    if channel == 1:
        # rgb --> Y (gray)
        imgIn = np.sum(imgIn * np.reshape([65.481, 128.553, 24.966], [1, 1, 1, 3]) / 255.0, axis=3,
                       keepdims=True) + 16.0

    # to Tensor
    ts = (3, 0, 1, 2)  ############# dimension order should be [C, T, H, W]
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)

    # normalization [-1,1]
    # imgIn = (imgIn / 255.0 - 0.5) * 2

    # normalization [0,1]
    imgIn = (imgIn / 255.0)
    
    return imgIn