# 模型统计 

## 依赖

```shell
# 用于统计计算量、参数量等
pip install ptflops 
```

## 使用

`main.py`仅配置Config()参数

| 参数        | 备注                        | 举例                             |
| ----------- | --------------------------- | -------------------------------- |
| input_size  | 图像大小[batch,channel,h,w] | [1,3,224, 224]                   |
| platform    | 测试平台 支持CPU、单GPU     | {'cpu'}    or  {'cpu', 'cuda:1'} |
| model       | 网络模型                    | resnet18()                       |
| warmup_nums | 预热次数                    | 100                              |
| iter_nums   | 计算耗时均值时的迭代次数    | 600                              |

## 结果

| img          | model    | FLOPs | GPU  | params | speed(CPU/GPU) |
| ------------ | -------- | ----- | ---- | ------ | -------------- |
| [3,224,224]] | resnet18 | 1.82G | 977M | 11.69M | 136.12/3.43 ms |
