from paddle.vision.transforms import functional as F
import paddle.nn as nn
from model import SiameseNet
from PIL import Image
import paddle
import sys
from model import ResNet

try:
    src_path = sys.argv[1]
except:
    print("请输入需要匹配的图片路径！")
    exit(-1)

model = "basic"

if model == "resnet":
    size = 100
    net = ResNet()
elif model == "basic":
    size = 100
    net = SiameseNet()

net.eval()
paddle.set_device("gpu:0")


layer_state_dict = paddle.load(model + "_acc38.pdparams")
net.set_state_dict(layer_state_dict)
dist = nn.PairwiseDistance(keepdim=True)

def load_img(path1,path2):
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    img1 = img1.convert("L")
    img2 = img2.convert("L")
    img1 = img1.resize((size, size))
    img2 = img2.resize((size, size))
    return F.to_tensor(img1),F.to_tensor(img2)

min_distance = 10000
clas = None
with open("database.txt","r") as r:
    lines = r.readlines()
    for line in lines:
        pic,label = line.replace("\n","").split("\t")
        image1,image2 = load_img(src_path,pic)
        image1 = paddle.reshape(image1,[1,1,size,size])
        image2 = paddle.reshape(image2,[1,1,size,size])
        output1, output2 = net(image1, image2)
        euclidean_distance = dist(output1,output2)
        if min_distance > euclidean_distance:
            clas = label
            min_distance = euclidean_distance
        
print("The picture is similar with {}, the euclidean_distance is {}".format(clas,min_distance.numpy()[0][0]))