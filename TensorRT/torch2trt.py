import torch
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet

# ================初始化Pytorch模型及输入===============
model = alexnet(pretrained=True).eval().cuda()
x = torch.ones((1, 3, 224, 224)).cuda()
y_torch = model(x)


# ================转化TensorRT并保存====================
# Pytorch转为TensorRT模型
model_trt = torch2trt(model, [x])
# 保存TensorRT模型权重
torch.save(model_trt.state_dict(), 'alexnet_trt.pth')


# ================加载模型并推理=========================
from torch2trt import TRTModule
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('alexnet_trt.pth'))
y_trt = model_trt(x)


# ================验证输出==============================
print(torch.max(torch.abs(y_torch - y_trt)))