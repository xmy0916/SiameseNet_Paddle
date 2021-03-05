import paddle.nn as nn
import paddle

class ContrastiveLoss(nn.Layer):
  def __init__(self,margin=2.0):
    super(ContrastiveLoss,self).__init__()
    self.margin = margin

  def forward(self,output1,output2,label):
    dist = nn.PairwiseDistance(keepdim=True)
    euclidean_distance = dist(output1,output2)
    loss_contrastive = \
            paddle.mean((1-label) * \
            paddle.pow(euclidean_distance, 2) + \
            (label) * \
            paddle.pow( \
                paddle.clip(self.margin - euclidean_distance, min=0.0), 2) \
            )
    return loss_contrastive

