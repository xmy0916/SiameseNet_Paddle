import os
import random
is_train = True

if not is_train:
    total_num = 100
    data_path = "/home/aistudio/work/data/faces/training/"
    if os.path.exists("eval_list.txt"):
        os.remove("/home/aistudio/work/eval_list.txt")
    w = open("eval_list.txt","w")
else:
    total_num = 500000
    data_path = "/home/aistudio/work/data/faces/training/"
    if os.path.exists("train_list.txt"):
        os.remove("/home/aistudio/work/train_list.txt")
    w = open("train_list.txt","w")


cls = []
for root,dir,file in os.walk(data_path):
  cls.append(dir)
    
cls = cls[0]



#print(cls)
for i in range(total_num):
  is_same = random.choice([True,False])
  cls_1 = random.choice(cls)
  if is_same:
    cls_2 = cls_1
  else:
    cls_2 = random.choice(cls)
    while cls_2 == cls_1:
      cls_2 = random.choice(cls)
  pic_1_path = os.path.join(data_path,cls_1)
  pic_2_path = os.path.join(data_path,cls_2)
  pic_1_list = os.listdir(pic_1_path)
  pic_2_list = os.listdir(pic_2_path)
  pic_1 = random.choice(pic_1_list)
  pic_2 = random.choice(pic_2_list)
  while pic_1.split(".")[-1] not in ["jpg","png","pgm"]:
    pic_1 = random.choice(pic_1_list)
  while pic_2.split(".")[-1] not in ["jpg","png","pgm"]:
    pic_2 = random.choice(pic_2_list)
  
  if is_train:
    w.write(os.path.join(pic_1_path,pic_1)+"\t"+os.path.join(pic_2_path,pic_2)+"\t"+str(int(is_same))+"\n")
  else:
    w.write(os.path.join(pic_1_path,pic_1)+"\t"+cls_1+"\n")
  
w.close()
