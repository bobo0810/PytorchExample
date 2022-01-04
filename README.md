# PytorchGuide

#### 收录到[PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)

### [AMP](./AMP/README.md)

- 自动混合精度训练

### [DDP](./DDP/readme.md)

- 分布式数据并行（多机多卡）

### [模型统计](./ModelAnalysis/readme.md)

- 参数量|计算量|GPU占用|耗时(CPU/GPU)

### [移动端部署](./ModelConver/readme.md)

- Pytorch->ONNX-> NCNN / MNN

### [TensorRT最佳实践](./TensorRT/readme.md)

- TensorRT API
- Pytorch->ONNX->TensorRT

### [数据相关](https://github.com/bobo0810/PytorchGuide/tree/main/DataTools)

- YOLO->VOC 标签格式转化
- YOLO 结果可视化 

### 小工具

- Pytest: 测试框架 

```bat
pytest  test.py             //自动运行 命名包含"test"的函数
pytest  test.py::test_hello //指定运行某个函数
pytest  -s test.py          // 同时显示测试函数中print()输出
```

- Fire: 自动生成命令行接口，方便运行指定函数