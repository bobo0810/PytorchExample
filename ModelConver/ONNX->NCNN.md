## 1. ONNX->NCNN

示例库: [Face-Detector-1MB-with-landmark](https://github.com/biubug6/Face-Detector-1MB-with-landmark)

### 编译NCNN

1. 安装Homebrew

   ```shell
   # macOS10.15.7
   /bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"
   ```

2. 安装第三方依赖

   ```shell
   brew install cmake
   brew install protobuf
   brew install opencv  //自动安装很多依赖  默认安装路径/usr/local/Cellar/opencv
   ```

3. 编译NCNN

   ```shell
   cd <ncnn-root-dir>   //进入ncnn根路径
   mkdir -p build
   cd build
   cmake .. #生成Makefile文件
   make  #根据Makefile文件进行编译
   make install #生成install文件夹
   ```

### 转化

1. 转为ncnn

   ```shell
   cd <ncnn-root-dir>/build/tools/onnx
   ./onnx2ncnn  faceDetector_sim.onnx   face.param  face.bin
   ```

### C++推理

1. 验证输出

(1)将`<ncnn-root-dir>/build/install`的文件替换到`Face-Detector-1MB-with-landmark/Face_Detector_ncnn/ncnn`目录下

(2)将 face.param 和face.bin移动到`Face_Detector_ncnn/model`目录下

(3)`Face_Detector_ncnn/FaceDetector.cpp` 第53、56、59行key分别更改为"bbox","prob","landmark"，与转化前key值对应。

(4)`Face_Detector_ncnn/main.cpp` 若使用retinaface模型，应将false->true。

(5)可选项

  -  anchor比例：`FaceDetector.cpp`第202行
  -  图像尺寸：`main.cpp`第27行

2. 构建项目

在`Face_Detector_ncnn/CMakeLists.txt`设置opencv路径

```shell
set(OpenCV_DIR "/usr/local/Cellar/opencv/4.5.2/")
```

编辑将出现如下错误

```shell
cmake .. 
make -j4 #开4个线程进行编译

#将出现错误 include未找到opencv2
fatal error: 'opencv2/opencv.hpp' file not found
fatal error: 'opencv2/core/core.hpp' file not found

# 原因
# opencv2的include路径为/usr/local/Cellar/opencv/4.5.2/include/opencv4/
```

3. 解决

```cmake
# CMakeLists文件修改两个路径
19行： ${OpenCV_DIR}/include  = /usr/local/Cellar/opencv/4.5.2/include/opencv4/
22行： ${OpenCV_DIR}/lib      = /usr/local/Cellar/opencv/4.5.2/lib
```

```shell
#编译
cmake .. 
make -j4
# 生成可执行文件FaceDetector
# 验证
./FaceDetector
```

![avatar](./imgs/ncnn.jpeg)



## 2. NCNN优化

作用：（1）优化模型，融合算子 （2）FP32->FP16

```shell
cd <ncnn-root-dir>/build/tools/
# flag:0为FP32，1为FP16
./ncnnoptimize  ncnn.param  ncnn.bin  new.param  new.bin flag
```



**参考**

[macOS编译NCNN](https://www.bilibili.com/read/cv10224407/)

[NCNN](https://github.com/Tencent/ncnn)

[NCNN深度学习框架之Optimize优化器](https://www.cnblogs.com/wanggangtao/p/11313705.html)

