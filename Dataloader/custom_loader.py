import cv2, copy, logging, os
import numpy as np

import utils
from glob import glob
import os.path as path
import torchvision.transforms as transforms

from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, cfg, root, transform=None, givenGlob='*/*.jpg'):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.root = root
        self.number_landmarks = cfg.WFLW.NUM_POINT
        self.Fraction = cfg.WFLW.FRACTION

        self.Transform = transform

        self.filelist = glob(path.join(root, givenGlob), recursive=True)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        path = self.filelist[idx]

        Img = cv2.imread(path)
        Img_shape = Img.shape

        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)

        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)

        height, width, channels = Img.shape
        BBox = np.array([0, 0, width, height])

        input = cv2.resize(Img, (self.Image_size, self.Image_size), interpolation = cv2.INTER_AREA)

        if self.Transform is not None:
            input = self.Transform(input)

        return input, path, BBox