# 服务端部署

TensorRT仅适用`Nvidia GPU`

## 模型转化

### 方案1

- 环境： (1)部署TensorRT  (2)安装torch2trt

- 通过torch2trt可直接转为TensorRT。经测试，推理加速一倍左右，效果因模型而异。

### 方案2

- Pytorch->ONNX->TensorRT

- 待更

## 流程

![avatar](./imgs/conver2trt.svg)



## 参考

[TensorRT部署](http://zengzeyu.com/2020/07/09/tensorrt_01_installation/)

[TensorRT部署常见错误](https://blog.csdn.net/QFJIZHI/article/details/107335865)

[TensorRT加速Pytorch](https://blog.csdn.net/leviopku/article/details/112963733)

[torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)



