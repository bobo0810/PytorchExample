import sys
import glob
import torch
import time
import os
def add_project_path():
    rootpath =os.path.abspath(os.path.dirname(__file__))
    sys.path.append(rootpath)
    sys.path.extend(glob.glob(rootpath+'/*'))
add_project_path()
from torchvision.models import resnet18
from ptflops import get_model_complexity_info
class Config():
    '''
    配置参数
    '''
    input_size = [1,3,224, 224] # 图像大小[batch,channel,h,w]
    platform = {'cpu','cuda:1'}   # 测试平台 支持CPU、单GPU
    model=resnet18() # 网络
    warmup_nums = 100 # 预热迭代次数
    iter_nums=600  # 计算耗时均值时的迭代次数



def main():
    cfg=Config()

    # 推理速度
    cal_time(cfg)

    # FLOPs、参数量
    flops, params = get_model_complexity_info(cfg.model, (cfg.input_size[1], cfg.input_size[2], cfg.input_size[3]), as_strings=True,
                                              print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # 显存占用
    print('GPU memory: please use the command：nvidia-smi')


def cal_time(cfg):
    '''
    统计 CPU/GPU推理速度
    '''
    for platform in cfg.platform:
        if 'cuda' in  platform:
            os.environ['CUDA_VISIBLE_DEVICES'] = platform.replace("cuda:", "")
            device = torch.device('cuda:0')

        else:
            device = torch.device(str(platform))
        input_size=cfg.input_size
        model=cfg.model
        iter_nums=cfg.iter_nums         # 计算耗时均值时的迭代次数
        warmup_nums=cfg.warmup_nums     # 预热迭代次数

        dump_input = torch.ones(input_size).to(device)
        model.to(device).eval()
        with torch.no_grad():
            # 预热
            for _ in range(warmup_nums):
                model(dump_input)
                if 'cuda' in platform:
                    torch.cuda.synchronize()
            # 正面
            start = time.time()
            for _ in range(iter_nums):
                model(dump_input)
                # 每次推理，均同步一次。算均值
                if 'cuda' in platform:
                    torch.cuda.synchronize()
            end = time.time()
            total_time=((end - start) * 1000)/float(iter_nums)
            print(str(platform) +' is  %.2f ms/img'%total_time)


if __name__ == '__main__':
    main()