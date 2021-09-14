import cv2
import  glob
from tqdm import tqdm
import os
cur_path=os.path.abspath(os.path.dirname(__file__))



class Config():
    '''
    Yolo -> VOC
    '''
   
    # images  test.jpg
    # labels  test.txt   
    #     类别   中心点x  中心点y   宽   高 
    #      2     0.44    0.37   0.68  0.12
    yolo_path="yolo_data/"   # yolo数据集根路径,存在images、labels两个文件夹
    save_path="VOC_label/"   #VOC格式xml文件保存路径
    class_name=["person","cat","dog"] # yolo类别序号对应的类别名称

def write_xml(save_path,txt_info):
    '''
    将信息写入xml文件
    save_path：保存路径
    txt_info：标签信息
    '''
    # 判断文件是否存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    suffix=txt_info['img_name'].split(".")[-1]
    xml_path=save_path+txt_info['img_name'].replace("."+suffix, ".xml")
    
    # xml写入基本信息      
    xml_file = open((xml_path), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('    <filename>' + txt_info['img_name'] + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(txt_info['img_W']) + '</width>\n')
    xml_file.write('        <height>' + str(txt_info['img_H']) + '</height>\n')
    xml_file.write('        <depth>' + str(txt_info['img_C']) + '</depth>\n')
    xml_file.write('    </size>\n')
    
    # xml写入bbox   eg:[class_index,xmin,ymin,xmax,ymax])
    for bbox in txt_info["bbox"]:
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + str(bbox[0]) + '</name>\n') #类别下标
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(bbox[1]) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(bbox[2]) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(bbox[3]) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(bbox[4]) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')
    xml_file.write('</annotation>')
    xml_file.close()

def Yolo2VOC(cfg):
    '''
    Yolo格式转化为VOC格式
    '''
    imgs_list=glob.glob(cfg.yolo_path+"images/*")

    # n=9000
    # imgs_list=[imgs_list[i:i + n] for i in range(0, len(imgs_list), n)]

    # 遍历imgs
    for i in tqdm(range(len(imgs_list))):
        img_path=imgs_list[i]
        suffix=img_path.split("/")[-1].split(".")[-1]
        txt_path=img_path.replace("/images/","/labels/").replace("."+suffix, ".txt")
       
        # 读取txt
        txt_info={}
        txt_info["img_name"]=img_path.split("/")[-1]
        txt_info["bbox"]=[]
        
        # 读取图片
        H,W,C=cv2.imread(img_path).shape 
        txt_info["img_H"],txt_info["img_W"],txt_info["img_C"]=H,W,C
        

        # 获取该照片对应所有bbox
        txt_file = open(txt_path,"r")
        for line in txt_file.readlines():
            line = line.strip()
            class_index,x,y,w,h = map(float,line.strip().split(' '))
            # 转为bbox左上角、右下角 
            xmin = int((x - w / 2 ) * W)
            ymin = int((y - h / 2) * H)
            xmax = int((x + w / 2) * W)
            ymax = int((y + h / 2) * H)
            # im_ = cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(0,255,0),2) #可视化


            # 下标替换为真实类别
            class_name=cfg.class_name[int(class_index)]
            txt_info["bbox"].append([class_name,xmin,ymin,xmax,ymax])
        
        write_xml(cfg.save_path,txt_info)
        
if __name__ == "__main__":
    cfg=Config()

    # Yolo -> VOC
    Yolo2VOC(cfg)