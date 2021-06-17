## Pytorch->ONNX

示例库: [Face-Detector-1MB-with-landmark](https://github.com/biubug6/Face-Detector-1MB-with-landmark)

1. 验证输出

   convert_to_onnx.py  

   ```python
   # RetinaFace网络输出三个参数：bbox、类别置信度、关键点
   output_names = ["output0"] 改为 output_names = ["bbox","prob","landmark"]
   ```

2. 转为ONNX

   注意： <font color=red>opset_version=11 与ONNX瘦身共用将导致推理异常</font>

   ```shell
   # 生成faceDetector.onnx 
   python convert_to_onnx.py --trained_model ./weights/mobilenet0.25_Final.pth   --network mobile0.25
   ```

3. ONNX瘦身

   ```shell
   #安装onnx-simplifier
   pip3 install -U pip && pip3 install onnx-simplifier
   # 生成faceDetector_sim.onnx
   python3 -m onnxsim faceDetector.onnx faceDetector_sim.onnx
   ```

**参考**

[onnx-simplifier](https://github.com/daquexian/onnx-simplifier)

