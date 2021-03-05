# 基于siamese net实现人脸比对
参考项目：

[Facial-Similarity-with-Siamese-Networks-in-Pytorch](https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch)

[SiameseNet_face_compare](https://github1s.com/xmy0916/SiameseNet_face_compare/)

用处：
- 该项目可以运用在将一张人脸图和数据库中的所有人脸图比较判断数据库中是否有这个人，在人脸识别开门的场景使用。
- 在分类问题中，只需要一张图片就可以进行分类的训练，建立好所有数据的数据库后用需要分类的图和数据库中所有图比较，特征向量距离最小的可以视为同一类别


注!!!!：加了resnet也不太行，可能人脸相似度太高了，目前测试最好的效果38/100，太差了...估计要加特征点信息才能有效，所以就看看了解思路就好了。

# 创建数据集
```bash
python3 make_dataset.py # 数据集的数量可以进代码修改
```

# 训练
```bash
python3 train.py # 会保存~/work/SiameseNet.pdparams参数文件
```

# 验证
```bash
python3 eval.py
```

# 建立数据库
```bash
python3 make_database.py
```

# 数据库搜索
```bash
python3 find_in_database.py data/faces/training/s19/1.pgm
```
