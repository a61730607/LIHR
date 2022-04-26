
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

feats = torch.randn((8,512,128,256))
probs = torch.randn((8,19,128,256))
batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
probs = probs.view(batch_size, c, -1)
print(probs.size())
feats = feats.view(batch_size, feats.size(1), -1)
feats = feats.permute(0, 2, 1) # batch x hw x c 
print(feats.size())
probs = F.softmax(1 * probs, dim=2)# batch x k x hw
print(probs.size())
ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
print(ocr_context.size())