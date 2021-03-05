import paddle
from paddle.io import Dataset, DataLoader
from data_reader import MyDataset
from model import SiameseNet
from paddle.vision.transforms import functional as F
import paddle.nn as nn
from PIL import Image


net = SiameseNet()
net.eval()
paddle.set_device("gpu:0")


layer_state_dict = paddle.load("basic_acc38.pdparams")
net.set_state_dict(layer_state_dict)
dist = nn.PairwiseDistance(keepdim=True)

def load_img(path1,path2):
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    img1 = img1.convert("L")
    img2 = img2.convert("L")
    img1 = img1.resize((100, 100))
    img2 = img2.resize((100, 100))
    return F.to_tensor(img1),F.to_tensor(img2)

with open("train_list.txt","r") as r:
    lines = r.readlines()
    index = 0
    for line in lines:
        index += 1
        path1,path2,label = line.replace("\n","").split("\t")
        image1,image2 = load_img(path1,path2)
        image1 = paddle.reshape(image1,[1,1,100,100])
        image2 = paddle.reshape(image2,[1,1,100,100])
        output1, output2 = net(image1, image2)
        
        euclidean_distance = dist(output1,output2)
        print("label is {} \t distance is {}".format(label,euclidean_distance.numpy()[0][0]))
        if index > 10:
            break


