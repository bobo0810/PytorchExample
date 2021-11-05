import onnx
import onnxruntime
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import time
import torchvision
import numpy as np
import os
import sys
current_path=os.path.abspath(os.path.dirname(__file__))
'''
封装通用代码
'''

def Init_TensorRT(trt_path):
    '''
    初始化TensorRT引擎
    trt_path: trt文件
    '''
    # 加载cuda引擎
    engine = load_engine(trt_path)
    # 创建CudaEngine之后,需要将该引擎应用到不同的卡上配置执行环境
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)  # input, output: host # bindings
    return [context,inputs, outputs, bindings, stream]
def load_engine(trt_path):
    """
    加载cuda引擎
    trt_path: TensorRT引擎文件
    """
    # 以trt的Logger为参数，使用builder创建计算图类型INetworkDefinition
    TRT_LOGGER = trt.Logger()

    # 如果已经存在序列化之后的引擎，则直接反序列化得到cudaEngine
    if os.path.exists(trt_path):
        print("Reading engine from file: {}".format(trt_path))
        with open(trt_path, 'rb') as f, \
                trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())  # 反序列化
    else:
        print('No Found:'+trt_path)
        raise FileNotFoundError


def allocate_buffers(engine):
    '''
    TRT分配缓存
    '''
    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            """
            host_mem: cpu memory
            device_mem: gpu memory
            """
            self.host = host_mem     # 主机数据
            self.device = device_mem # GPU数据

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        # print(binding) # 绑定的输入输出
        # print(engine.get_binding_shape(binding)) # get_binding_shape 是变量的大小
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        # volume 计算可迭代变量的空间，指元素个数
        # size = trt.volume(engine.get_binding_shape(binding)) # 如果采用固定bs的onnx，则采用该句
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # get_binding_dtype  获得binding的数据类型
        # nptype等价于numpy中的dtype，即数据类型
        # allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)  # 创建锁业内存
        device_mem = cuda.mem_alloc(host_mem.nbytes)  # cuda分配空间
        # print(int(device_mem)) # binding在计算图中的缓冲地址
        bindings.append(int(device_mem))
        # append to the appropriate list
        if engine.binding_is_input(binding): 
            inputs.append(HostDeviceMem(host_mem, device_mem)) # 绑定输入
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem)) # 绑定输出
    return inputs, outputs, bindings, stream


def Do_Inference(context, bindings, inputs, outputs, stream):
    '''
    执行推理
    '''
    # htod：host to device 将数据由主机迁移到gpu device
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs] 
   
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # dtoh：device to host 
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
 
    # Synchronize the stream 同步流后才能得到预测结果
    stream.synchronize()

    # 返回预测结果 一维numpy
    return [out.host for out in outputs] 


def Torch_to_ONNX(net,input_size,onnx_path,device):
    '''
    torch->onnx(仅支持固定输入尺度)
    input_size: 输入尺度 [N,3,224,224] 
    onnx_path: onnx权重文件的保存路径
    device: "cuda:0"
    '''
    net.to(device)
    net.eval()
    # 转为ONNX
    torch.onnx.export(net,  # 待转换的网络模型和参数
                    torch.randn(tuple(input_size), device=device),  # 虚拟的输入，用于确定输入尺寸和推理计算图每个节点的尺寸
                    onnx_path,  # 输出文件路径
                    verbose=False,  # 是否以字符串的形式显示计算图
                    input_names=["input"], 
                    output_names=["output"],  # 输出节点的名称
                    opset_version=13,  # onnx支持算子的版本
                    do_constant_folding=True,  # 是否压缩常量
                    )


    # 验证模型
    net = onnx.load(onnx_path)  # 加载onnx 计算图
    onnx.checker.check_model(net)  # 检查文件模型是否正确
    onnx.helper.printable_graph(net.graph)  # 输出onnx的计算图

    # ONNX推理
    session = onnxruntime.InferenceSession(onnx_path)  # 创建一个运行session，类似于tensorflow
    output = session.run(None, {"input": np.random.rand(input_size[0],input_size[1], input_size[2], input_size[3]).astype('float32')})  # 输入必须是numpy类型

    print('ONNX file in ' + onnx_path)
    print('============Pytorch->ONNX SUCCESS============')


def ONNX_to_TensorRT(fp16_mode=False,onnx_path=None,trt_path=None,max_batch_size=1):
    """
    生成cudaEngine，并保存引擎文件(仅支持固定输入尺度)
    
    max_batch_size: 默认为1，不支持动态batch
    fp16_mode: True则fp16预测
    onnx_path: 将加载的onnx权重路径
    trt_path: trt引擎文件保存路径
    """
    # 通过logger报告错误、警告、信息
    TRT_LOGGER = trt.Logger()

    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser: 
        builder.max_workspace_size = 1 << 30  # 预先分配的工作空间大小,即ICudaEngine执行时GPU最大需要的空间
        builder.max_batch_size = max_batch_size  # 执行时最大可以使用的batchsize
        builder.fp16_mode = fp16_mode 

        # ########解析onnx文件，填充计算图#########
        if not os.path.exists(onnx_path):
            quit("ONNX file {} not found!".format(onnx_path))
        print('loading onnx file from path {} ...'.format(onnx_path))
        with open(onnx_path, 'rb') as model: 
            print("Begining onnx file parsing")
            parser.parse(model.read())  # OnnxParser解析onnx文件，为network对象构建网络并填充权重
        print("Completed parsing of onnx file")
    
        ########builder基于计算图创建引擎#########
        print("Building an engine from file{}' this may take a while...".format(onnx_path))
        output_shape=network.get_layer(network.num_layers - 1).get_output(0).shape # 查看最后一层网络输出尺寸
        # network.mark_output(network.get_layer(network.num_layers -1).get_output(0)) #设置输出
        engine = builder.build_cuda_engine(network)  # 构建引擎
        print("Completed creating Engine")

        # 保存engine供以后直接加载使用
        with open(trt_path, 'wb') as f:
            f.write(engine.serialize())  # 序列化

        print('TensorRT file in ' + trt_path)
        print('============ONNX->TensorRT SUCCESS============')