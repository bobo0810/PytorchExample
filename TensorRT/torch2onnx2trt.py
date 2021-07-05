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


class ONNX_Config():
    '''
    ONNX参数
    '''
    input_size=[1,3,224,224] # 输入尺寸
    device_id="cuda:2" # "cpu" or "cuda:2"
    save_onnx_name=current_path+"/resnet18.onnx" # 保存onnx模型的路径

class TensorRT_Config():
    '''
    TensorRT参数
    '''
    max_batch_size = 1  # 输入模型的batch数
    input_size = [max_batch_size, 3, 224, 224]  # 输入尺寸
    output_size= (max_batch_size,1000) #输出尺寸 [batch_size,class_nums]
    onnx_model_path =current_path+"/resnet18.onnx"  # 加载onnx模型的路径
    # 模型相关
    fp16_mode = False     # 是否支持FP16 依赖硬件
    trt_engine_path = current_path+"/model_fp16_{}.trt".format(fp16_mode) # 保存TensorRT模型的路径

def torch2onnx(net,onnx_cfg,device):
    # 转为ONNX
    torch.onnx.export(net,  # 待转换的网络模型和参数
                      torch.randn(tuple(onnx_cfg.input_size), device=device),  # 虚拟的输入，用于确定输入尺寸和推理计算图每个节点的尺寸
                      onnx_cfg.save_onnx_name,  # 输出文件路径
                      verbose=False,  # 是否以字符串的形式显示计算图
                      input_names=["input"] + ["params_%d" % i for i in range(120)],  # 输入节点的名称，这里也可以给一个list，list中名称分别对应每一层可学习的参数，便于后续查询
                      output_names=["output"],  # 输出节点的名称
                      opset_version=10,  # onnx 支持采用的operator set, 应该和pytorch版本相关，目前我这里最高支持10
                      do_constant_folding=True,  # 是否压缩常量
                      # dynamic_axes={"input":{0: "batch_size", 2: "h"}, "output":{0: "batch_size"},} #设置动态维度，此处指明input节点的第0维度可变，命名为batch_size
                      # 经测试，采用dynamic_axes，ONNX动态维度OK，而TensorRT运行出错
                      )


    # 验证模型
    # import onnx  # 注意这里导入onnx时必须在torch导入之前，否则会出现segmentation fault
    net = onnx.load(onnx_cfg.save_onnx_name)  # 加载onnx 计算图
    onnx.checker.check_model(net)  # 检查文件模型是否正确
    onnx.helper.printable_graph(net.graph)  # 输出onnx的计算图

    # ONNX推理
    import onnxruntime
    session = onnxruntime.InferenceSession(onnx_cfg.save_onnx_name)  # 创建一个运行session，类似于tensorflow
    out_r = session.run(None, {"input": np.random.rand(onnx_cfg.input_size[0], onnx_cfg.input_size[1], onnx_cfg.input_size[2], onnx_cfg.input_size[3]).astype('float32')})  # 模型运行，注意这里的输入必须是numpy类型

    print('ONNX file in ' + onnx_cfg.save_onnx_name)
    print('============Pytorch->ONNX SUCCESS============')

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

def onnx2tensorrt(trt_cfg):
    """
    生成cudaEngine，并保存引擎文件
    """
    # 以trt的Logger为参数，使用builder创建计算图类型INetworkDefinition
    TRT_LOGGER = trt.Logger()

    # 由onnx创建cudaEngine
    # 使用logger创建一个builder
    # builder创建一个计算图 INetworkDefinition
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # In TensorRT 7.0, the ONNX parser only supports full-dimensions mode, meaning that your network definition must be created with the explicitBatch flag set. For more information, see Working With Dynamic Shapes.

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:  # 使用onnx的解析器绑定计算图，后续将通过解析填充计算图
        builder.max_workspace_size = 1 << 30  # 预先分配的工作空间大小,即ICudaEngine执行时GPU最大需要的空间
        builder.max_batch_size = trt_cfg.max_batch_size  # 执行时最大可以使用的batchsize
        builder.fp16_mode = trt_cfg.fp16_mode

        # 解析onnx文件，填充计算图
        if not os.path.exists(trt_cfg.onnx_model_path):
            quit("ONNX file {} not found!".format(trt_cfg.onnx_model_path))
        print('loading onnx file from path {} ...'.format(trt_cfg.onnx_model_path))
        with open(trt_cfg.onnx_model_path, 'rb') as model:  # 二值化的网络结果和参数
            print("Begining onnx file parsing")
            parser.parse(model.read())  # 解析onnx文件
        # parser.parse_from_file(trt_cfg.onnx_model_path) # parser还有一个从文件解析onnx的方法

        print("Completed parsing of onnx file")
        # 填充计算图完成后，则使用builder从计算图中创建CudaEngine
        print("Building an engine from file{}' this may take a while...".format(trt_cfg.onnx_model_path))

        #################
        output_shape=network.get_layer(network.num_layers - 1).get_output(0).shape
        # network.mark_output(network.get_layer(network.num_layers -1).get_output(0))
        engine = builder.build_cuda_engine(network)  # 注意，这里的network是INetworkDefinition类型，即填充后的计算图
        print("Completed creating Engine")

        # 保存engine供以后直接反序列化使用
        with open(trt_cfg.trt_engine_path, 'wb') as f:
            f.write(engine.serialize())  # 序列化

        print('TensorRT file in ' + trt_cfg.trt_engine_path)
        print('============ONNX->TensorRT SUCCESS============')


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """
        host_mem: cpu memory
        device_mem: gpu memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
def allocate_buffers(engine):
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
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # htod： host to device 将数据由cpu复制到gpu device
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # 当创建network时显式指定了batchsize， 则使用execute_async_v2, 否则使用execute_async
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # gpu to cpu
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def trt_inference(trt_path,img_np_nchw,shape_of_output):
    '''
    TensorRT推理
    :param trt_path:
    :param img_np_nchw: 输入图像
    :param shape_of_output: 输出形状 (batch_size,class_nums)
    '''
    # 加载cuda引擎
    engine = load_engine(trt_path)
    # 创建CudaEngine之后,需要将该引擎应用到不同的卡上配置执行环境
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)  # input, output: host # bindings

    # ==============推理=============
    # 加载输入数据到buffer
    inputs[0].host = img_np_nchw.reshape(-1) #输入形状转为一维，作为输入
    # inputs[1].host = ... for multiple input  对于多输入情况

    t1 = time.time()
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)  # numpy data
    t2 = time.time()
    feat = trt_outputs[0].reshape(*shape_of_output) # 输出形状由一维转为指定形状
    return feat,t1,t2

if __name__ == "__main__":

    # ============1.Pytorch->ONNX============
    onnx_cfg = ONNX_Config() #配置onnx转化参数
    device = torch.device(onnx_cfg.device_id)
    # 初始化Pytorch模型
    torch_net = torchvision.models.resnet18(pretrained=True).to(device)
    torch_net.eval()
    # 转为ONNX模型
    torch2onnx(torch_net,onnx_cfg,device)

    # ============2.ONNX->TensorRT============
    trt_cfg = TensorRT_Config() #配置tesnorrt转化参数
    onnx2tensorrt(trt_cfg)

    # ============3.验证结果============
    # 读取图像
    img_np_nchw = np.ones((trt_cfg.input_size[0], trt_cfg.input_size[1], trt_cfg.input_size[2], trt_cfg.input_size[3]),dtype=float).astype(np.float32)

    # TensorRT推理
    feat,t1,t2=trt_inference(trt_path=trt_cfg.trt_engine_path,img_np_nchw=img_np_nchw,shape_of_output=trt_cfg.output_size)
    print('TensorRT ok!')

    # Pytorch推理
    input_for_torch = torch.from_numpy(img_np_nchw).to(device)
    t3 = time.time()
    feat_2 = torch_net(input_for_torch)
    t4 = time.time()
    feat_2 = feat_2.cpu().data.numpy()
    print('Pytorch ok!')

    # 计算误差
    mse = np.mean((feat - feat_2) ** 2)
    print("Inference time with the TensorRT engine: {}".format(t2 - t1))
    print("Inference time with the PyTorch model: {}".format(t4 - t3))
    print('MSE Error = {}'.format(mse))

    print('All completed!')