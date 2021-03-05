import paddle
from paddle.io import Dataset, DataLoader
from data_reader import MyDataset
from model import SiameseNet,ResNet
from loss import ContrastiveLoss
from PIL import Image
from paddle.vision.transforms import functional as F
import paddle.nn as nn
from paddle.vision.models import LeNet
from paddle.vision.models.resnet import BottleneckBlock, BasicBlock

def load_img(path1,path2,size):
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    img1 = img1.convert("L")
    img2 = img2.convert("L")
    img1 = img1.resize((size, size))
    img2 = img2.resize((size, size))
    return F.to_tensor(img1),F.to_tensor(img2)

batch_size = 64
EPOCH_NUM = 1000
lr = "Exponential"
model = "basic" # basic
if model == "basic":
    size = 100
elif model == "resnet":
    size = 100 # 224好像跑不动...




if model == "basic":
    net = SiameseNet()
    dataset = MyDataset("/home/aistudio/work/train_list.txt",size)
elif model == "resnet":
    net = ResNet()
    dataset = MyDataset("/home/aistudio/work/train_list.txt",size)

loader = DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=2)

net.train()
if lr == "Cosine":
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.0005, T_max=10, verbose=False)
elif lr == "Exponential":
    scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.0005, gamma=0.9999, verbose=False)


opt = paddle.optimizer.Adam(learning_rate=scheduler,
                                parameters=net.parameters())
criterion = ContrastiveLoss()
best_acc = 0
dist = nn.PairwiseDistance(keepdim=True)
paddle.set_device("gpu:0")

with open("database.txt","r") as r:
    database = r.readlines()
with open("/home/aistudio/work/eval_list.txt","r") as r:
    test_file = r.readlines()

for epoch in range(EPOCH_NUM):
    sum_loss = 0.0
    index = 0
    for i, (image1, image2, label) in enumerate(loader()):
        index += 1
        output1, output2 = net(image1, image2) # 64,5

        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        if i % 10 == 0:
            print("epoch:{} iter:{} now loss:{} lr:{}".format(epoch,i,loss_contrastive.numpy()[0],scheduler.get_lr()))
        sum_loss += loss_contrastive.numpy()[0]
        opt.step()
        opt.clear_grad()
        #scheduler.step()

    ave_loss = sum_loss * 1.0 / index

    # 评估
    print("eval!")
    acc = 0
    for file in test_file:
        src_path,src_label = file.replace("\n","").split("\t")
        clas = ""
        min_distance = 100000
        for line in database:
            pic,label = line.replace("\n","").split("\t")
            image1,image2 = load_img(src_path,pic,size)
            image1 = paddle.reshape(image1,[1,1,size,size])
            image2 = paddle.reshape(image2,[1,1,size,size])
            output1, output2 = net(image1, image2) # 64,5
            euclidean_distance = dist(output1,output2)
            if min_distance > euclidean_distance:
                clas = label
                min_distance = euclidean_distance
        if src_label == clas:
            acc += 1
    print("epoch:{} acc:{} total:{} loss:{}".format(epoch,acc,len(test_file),loss_contrastive.numpy()[0]))
    if epoch == 0 or acc > best_acc:
        layer_state_dict = net.state_dict()
        paddle.save(layer_state_dict, model + "_acc{}.pdparams".format(acc))
        best_acc = acc
        print("save model！")

