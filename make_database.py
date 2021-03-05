import os
import random

dataPath = "/home/aistudio/work/data/faces/training/"
cls = []
for root,dir,file in os.walk(dataPath):
  cls.append(dir)
    
cls = cls[0]
with open("database.txt","w") as w:
    for c in cls:
        cls_path = os.path.join(dataPath,c)
        pic_name = random.choice(os.listdir(cls_path))
        w.write(os.path.join(cls_path,pic_name) + "\t" + c + "\n")

