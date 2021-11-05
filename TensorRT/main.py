import onnx
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import time
import torchvision
import numpy as np
import os
current_path=os.path.abspath(os.path.dirname(__file__))
from trt_com import Torch_to_ONNX,ONNX_to_TensorRT,Init_TensorRT,Do_Inference



batch_size=3 # 固定尺度 eg:1、6、8...
class ONNX_Config():
    '''
    ONNX参数
    '''
    input_size=[batch_size,3,224,224] # 输入尺寸
    device_id="cuda:0" 
    onnx_path=current_path+"/model.onnx" # onnx模型的保存路径

class TensorRT_Config():
    '''
    TensorRT参数
    '''
    output_size= [batch_size,1000] #输出尺寸  resnet18输出1000分类
    fp16_mode = True     # 是否支持FP16 依赖硬件
    trt_path = current_path+"/model_fp16_{}.trt".format(fp16_mode) # TRT引擎的保存路径

if __name__ == "__main__":
    # ============1.Pytorch->ONNX============
    onnx_cfg = ONNX_Config() #配置onnx转化参数
    device = torch.device(onnx_cfg.device_id)
    # 初始化Pytorch
    torch_net = torchvision.models.resnet18(pretrained=True).to(device)
    torch_net.eval()
    # 转为ONNX模型
    Torch_to_ONNX(torch_net,onnx_cfg.input_size,onnx_cfg.onnx_path,device)
    

    # ============2.ONNX->TensorRT============
    trt_cfg = TensorRT_Config() #配置tesnorrt转化参数
    ONNX_to_TensorRT(trt_cfg.fp16_mode,onnx_cfg.onnx_path,trt_cfg.trt_path)


    # ============3.Trt预测============
    img_np_nchw = np.ones(tuple(onnx_cfg.input_size),dtype=float).astype(np.float32) # 输入数据

    [context,inputs, outputs, bindings, stream] =Init_TensorRT(trt_cfg.trt_path) # 加载引擎
    inputs[0].host = img_np_nchw.reshape(-1) # 绑定输入数据  一维npy
    # inputs[1].host = ... 适用多个输入

    t0 = time.time()
    output=Do_Inference(context, bindings, inputs, outputs, stream) # list  若网络仅一个输出，则len=1
    t1 = time.time()
    output=output[0].reshape(*trt_cfg.output_size) # 一维npy 恢复为 指定输出尺寸

    # ============4.Torch预测============
    input = torch.from_numpy(img_np_nchw).to(device)
    t2 = time.time()
    output_torch = torch_net(input)
    t3 = time.time()
    
    # ============5.计算误差============
    mse = np.mean((output - output_torch.cpu().detach().numpy()) ** 2)

    print('MSE Error = {}'.format(mse))
    print("Inference time with the TensorRT engine: {}".format(t1 - t0))
    print("Inference time with the PyTorch model: {}".format(t3 - t2))
    print('All completed!')
    