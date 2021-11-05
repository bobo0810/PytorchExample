'''
https://github.com/wang-xinyu/tensorrtx  lenet最简单示例
'''

import argparse
import os
import struct
import sys

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

INPUT_H = 32 #输入尺寸
INPUT_W = 32
OUTPUT_SIZE = 10 #输出形状  10分类
INPUT_BLOB_NAME = "data" # blob二进制对象  输入名称
OUTPUT_BLOB_NAME = "prob" # 输出名称

weight_path = "./lenet5.wts" # 二进制权重
engine_path = "./lenet5.engine" #trt引擎的保存路径

gLogger = trt.Logger(trt.Logger.INFO) # 通过logger报告错误、警告、信息（Builder/ICudaEngine/Runtime）


def load_weights(file):
    '''加载二进制权重文件'''
    print(f"Loading weights: {file}")

    assert os.path.exists(file), 'Unable to load weight file.'

    weight_map = {}
    with open(file, "r") as f:
        lines = [line.strip() for line in f]
    count = int(lines[0])
    assert count == len(lines) - 1
    for i in range(1, count + 1): # 遍历每行内容
        splits = lines[i].split(" ")
        name = splits[0] # 第一个值为网络名称
        cur_count = int(splits[1]) # 第二个值为 该行参数数量
        assert cur_count + 2 == len(splits) 
        values = [] #保存该行参数
        for j in range(2, len(splits)):
            # hex string to bytes to float
            values.append(struct.unpack(">f", bytes.fromhex(splits[j])))
        weight_map[name] = np.array(values, dtype=np.float32)

    return weight_map


def createLenetEngine(maxBatchSize, builder, config, dt):
    '''
    构建网络引擎
    dt: fp32 or fp16
    '''


    weight_map = load_weights(weight_path) # 加载二进制权重
    network = builder.create_network() # 创建网络对象

    data = network.add_input(INPUT_BLOB_NAME, dt, (1, INPUT_H, INPUT_W)) # 设置网络输入的名称和尺寸
    assert data
    # ============定义网络============
    # 定义卷积
    conv1 = network.add_convolution(input=data, # 输入tensor
                                    num_output_maps=6, # 输出通道
                                    kernel_shape=(5, 5), # 卷积核尺寸
                                    kernel=weight_map["conv1.weight"], # 赋值卷积核的权重[out_channels, in_channels, kernel_height, kernel_width]
                                    bias=weight_map["conv1.bias"]) # 赋值偏向权重[out_channels]
    assert conv1
    conv1.stride = (1, 1) # 设置卷积的步长
    
    # 定义激活函数
    relu1 = network.add_activation(conv1.get_output(0), # 前卷积层的输出
                                   type=trt.ActivationType.RELU)
    assert relu1
    
    # 定义池化
    pool1 = network.add_pooling(input=relu1.get_output(0),# 前激活层的输出
                                window_size=trt.DimsHW(2, 2), # 池化窗口大小
                                type=trt.PoolingType.AVERAGE) # 池化类型为平均池化
    assert pool1
    pool1.stride = (2, 2) # 池化步长

    conv2 = network.add_convolution(pool1.get_output(0), 16, trt.DimsHW(5, 5),
                                    weight_map["conv2.weight"],
                                    weight_map["conv2.bias"])
    assert conv2
    conv2.stride = (1, 1)

    relu2 = network.add_activation(conv2.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu2

    pool2 = network.add_pooling(input=relu2.get_output(0),
                                window_size=trt.DimsHW(2, 2),
                                type=trt.PoolingType.AVERAGE)
    assert pool2
    pool2.stride = (2, 2)

    # 定义全连接层
    fc1 = network.add_fully_connected(input=pool2.get_output(0),
                                      num_outputs=120,
                                      kernel=weight_map['fc1.weight'],
                                      bias=weight_map['fc1.bias'])
    assert fc1

    relu3 = network.add_activation(fc1.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu3

    fc2 = network.add_fully_connected(input=relu3.get_output(0),
                                      num_outputs=84,
                                      kernel=weight_map['fc2.weight'],
                                      bias=weight_map['fc2.bias'])
    assert fc2

    relu4 = network.add_activation(fc2.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu4

    fc3 = network.add_fully_connected(input=relu4.get_output(0),
                                      num_outputs=OUTPUT_SIZE,
                                      kernel=weight_map['fc3.weight'],
                                      bias=weight_map['fc3.bias'])
    assert fc3

    prob = network.add_softmax(fc3.get_output(0)) #经过softmax
    assert prob

    prob.get_output(0).name = OUTPUT_BLOB_NAME # 网络输出 赋值名称，便于后续通过名称拿出预测结果
    network.mark_output(prob.get_output(0)) # 将该tensor 标记为输出

    # Build engine
    builder.max_batch_size = maxBatchSize
    # builder.max_workspace_size = 1 << 20
    config.max_workspace_size= 1 << 20
    engine = builder.build_engine(network, config)

    del network
    del weight_map

    return engine


def APIToModel(maxBatchSize):
    '''将二进制权重转为trt引擎'''
    builder = trt.Builder(gLogger) # builder对象 用于推理
    config = builder.create_builder_config() # 为builder对象配置参数
    engine = createLenetEngine(maxBatchSize, builder, config, trt.float32)
    assert engine  # 断言引擎不为空

    # 保存为trt引擎文件
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    del engine
    del builder


def doInference(context, host_in, host_out, batchSize):
    '''
    trt推理
   
    host_in  输入数据
    host_out 空npy,用于接收输出
    '''
    engine = context.engine
    assert engine.num_bindings == 2 # 绑定的tensor数量  输入1+输出1

    devide_in = cuda.mem_alloc(host_in.nbytes) # cuda分配输入内存，返回“设备分配对象“地址
    devide_out = cuda.mem_alloc(host_out.nbytes)
    bindings = [int(devide_in), int(devide_out)] 
    stream = cuda.Stream() # 多个流 可以并行

    cuda.memcpy_htod_async(devide_in, host_in, stream) # 将主机内存的数据 复制到GPU上  htod即host_to_device
    context.execute_async(bindings=bindings, stream_handle=stream.handle) # 异步 GPU执行推理
    cuda.memcpy_dtoh_async(host_out, devide_out, stream)  # 将GPU数据 复制到主机内存上 dtoh即device_to_host
    stream.synchronize() #流同步后 host_out接收预测结果


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s",default=False, action='store_true') 
    parser.add_argument("-d", default=True, action='store_true')
    args = parser.parse_args()

    if not (args.s ^ args.d):
        print("arguments not right!")
        print("python lenet.py -s   # serialize model to plan file") # 将二进制权重转为trt引擎
        print("python lenet.py -d   # deserialize plan file and run inference") # 加载trt引擎并推理
        sys.exit()

    if args.s:
        APIToModel(1)
    else:
        runtime = trt.Runtime(gLogger) # 创建trt运行时，以便加载trt引擎
        assert runtime

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read()) # 加载trt引擎
        assert engine

        context = engine.create_execution_context() # 创建执行内容对象
        assert context

        data = np.ones((INPUT_H * INPUT_W), dtype=np.float32) # TRT输入为一维[1024]  1024=1*32*32
        host_in = cuda.pagelocked_empty(INPUT_H * INPUT_W, dtype=np.float32) # 页面锁定分配输入 dtype为输入数据类型  初始化一个与输入数据尺寸相同的空npy
        np.copyto(host_in, data.ravel()) # ravel将原数据拉伸为一维，不产生副本   copyto返回数组的副本，赋值给host_in    
        host_out = cuda.pagelocked_empty(OUTPUT_SIZE, dtype=np.float32) # 页面锁定分配输出  初始化一个与输出数据尺寸相同的空npy 
        doInference(context, host_in, host_out, 1) # 推理完成后 host_out保存结果

        print(f'Output: {host_out}')
