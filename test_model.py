import os
from re import U
import sys
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt

os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datetime import datetime
from model.unet import UNet2,UNet2_se,UNet2_Atten,UNet2_se_Atten,UNet2_se_Atten_Trans,UNet2_Atten_Trans
from model.SegNet import SegNet
from model.DDRNet import DualResNet,BasicBlock
from model.SmatUnet import SmaAt_UNet
from torch.utils.data import (DataLoader)
from model.utils import StrawDataset,load_and_cache_withlabel,PrintModelInfo,CaculateAcc,visual_result,CalculateMiou
BATCH_SIZE=1
PRETRAINED_MODEL_PATH="./output/output_model/SegNet_.pth"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
TF_ENABLE_ONEDNN_OPTS=0

"""dataset"""
val_type="test"
data_path_val=f"./dataset/src/{val_type}"
label_path_val=f"./dataset/label/{val_type}"
cached_file_val=f"./dataset/cache/{val_type}.pt"

def CreateDataloader(image_path,label_path,cached_file):
    features = load_and_cache_withlabel(image_path,label_path,cached_file,shuffle=False)  
    num_features = len(features)
    num_train = int(1* num_features)
    train_features = features[:num_train]
    dataset = StrawDataset(features=train_features,num_instances=num_train)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader

def visualize_segmentation(output, batch_index=0, class_names=None,filename=None):
    predicted = torch.argmax(output, dim=1)  # 形状为 [batch_size, height, width]
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()
    pred_image = predicted[batch_index]
    color_map = [
        [220, 20, 60],  # 秸秆（黄色）
        [153, 50, 204],  # 玉米（紫色）
        [127, 255, 0],    # 芝麻（绿色）
        [65, 105, 225],    # 背景（蓝色）
    ]

    colored_image = np.zeros((pred_image.shape[0], pred_image.shape[1], 3), dtype=np.uint8)
    for class_index, color in enumerate(color_map):
        colored_image[pred_image == class_index] = color

    # 使用 matplotlib 保存图像
    plt.figure(figsize=(8, 8))
    plt.imshow(colored_image)
    plt.axis('off')  # 不显示坐标轴
    if class_names:
        plt.title(f'Segmentation output for: {class_names[batch_index]}')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
       
def main():
    """Define Model"""
    #model=UNet2_se_Atten_Trans(3,4).to(DEVICE)
    model=SegNet().to(DEVICE)
    #model=DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=4, planes=32, spp_planes=128, head_planes=64, augment=False).to(DEVICE)
    #model=SmaAt_UNet(3,4).to(DEVICE)
    PrintModelInfo(model)
    """Pretrain"""
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH),strict=False)
    """Create dataloader"""
    dataloader_val=CreateDataloader(data_path_val,label_path_val,cached_file_val)
    """ validation """
    start_time=datetime.now()
    model.eval()
    sum_accuarcy=0
    sum_iou=0
    with open('record.txt', 'w') as record_file:
        with torch.no_grad():
            validation_iterator = tqdm.tqdm(dataloader_val, initial=0,desc="Iter", disable=False)
            for i, (image,label) in enumerate(validation_iterator):
                image,label= image.to(DEVICE),label.to(DEVICE) 
                output=model(image)
                output = output.reshape(output.shape[0], 4,256, 256)
                visualize_segmentation(output, batch_index=0, filename=f"./output/output_images/pred{i}")
                #visualize_segmentation(label, batch_index=0, filename=f"./output/output_images/label{i}")
                #visual_result(image[0],f"./output/output_images/original{i}")
                accuarcy=CaculateAcc(output,label)
                record_file.write(f"num {i} acc is {accuarcy}\n")
                iou=CalculateMiou(output,label,output.shape[1])
                sum_accuarcy=sum_accuarcy+ accuarcy
                sum_iou=sum_iou+iou
                validation_iterator.set_description('ValAcc= %3.3f %%,miou= %3.3f ' % (sum_accuarcy*100/(i+1),sum_iou/(i+1)))
    end_time=datetime.now()
    time=end_time-start_time
    print(f"inference use {time} s")
if __name__=="__main__":
    main()