import paddle
from paddle.io import Dataset, BatchSampler, DataLoader
import os
import numpy as np
from PIL import Image
import random
from paddle.vision.transforms import functional as F


class MyDataset(Dataset):
    def __init__(self, data_list,size):
        super(MyDataset, self).__init__()
        self.size = size
        with open(data_list,"r") as r:
            self.datalist = r.readlines()

    def __getitem__(self, index):
        data = random.choice(self.datalist)
        pic1,pic2,label = data.replace("\n","").split("\t")
        img1 = Image.open(pic1)
        img2 = Image.open(pic2)
        label = int(label)
        img1 = img1.convert("L")
        img2 = img2.convert("L")
        _img1 = img1.resize((self.size, self.size))
        _img2 = img2.resize((self.size, self.size))
        #return F.to_tensor(_img1),F.to_tensor(_img2),paddle.to_tensor(label,dtype="float32")
        return np.array(_img1)[np.newaxis,:,:].astype('float32'),np.array(_img2)[np.newaxis,:,:].astype('float32'),np.array([label]).astype('float32')

    def __len__(self):
        return len(self.datalist)



