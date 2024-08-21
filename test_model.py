import os
from re import U
import sys
import torch
import tqdm 
import numpy as np
import matplotlib.pyplot as plt

os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from model.unet import UNet
from torch.utils.data import (DataLoader)
from model.utils import StrawDataset,load_and_cache_withlabel,PrintModelInfo,CaculateAcc,visual_result
BATCH_SIZE=1
SAVE_MODEL='./output/output_model/'
PRETRAINED_MODEL_PATH="./output/output_model/UNet.pth"
Pretrain=False if PRETRAINED_MODEL_PATH ==" " else True
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
TF_ENABLE_ONEDNN_OPTS=0

"""dataset"""
val_type="test"
data_path_val=f"./dataset/src/val"
label_path_val=f"./dataset/label/val"
cached_file_val=f"./dataset/cache/{val_type}.pt"

def CreateDataloader(image_path,label_path,cached_file):
    features = load_and_cache_withlabel(image_path,label_path,cached_file,shuffle=True)  
    num_features = len(features)
    num_train = int(1* num_features)
    train_features = features[:num_train]
    dataset = StrawDataset(features=train_features,num_instances=num_train)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def visualize_segmentation(output, batch_index=0, class_names=None,filename=None):
    predicted = torch.argmax(output, dim=1)  # 形状为 [batch_size, height, width]
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()
    pred_image = predicted[batch_index]
    color_map = [
        [0, 0, 255],  # 秸秆（黄色）
        [128, 0, 128],  # 玉米（紫色）
        [0, 128, 0],    # 芝麻（绿色）
        [255, 255, 0],    # 背景（蓝色）
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
    model=UNet(3,4).to(DEVICE)
    PrintModelInfo(model)
    """Pretrain"""
    if Pretrain:
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH),strict=False)
    """Create dataloader"""
    dataloader_val=CreateDataloader(data_path_val,label_path_val,cached_file_val)
    """ validation """
    sum_accuarcy=0
    model.eval()
    with torch.no_grad():
        validation_iterator = tqdm.tqdm(dataloader_val, initial=0,desc="Iter", disable=False)
        for i, (image,label) in enumerate(validation_iterator):
            image,label= image.to(DEVICE),label.to(DEVICE) 
            output=model(image)
            output = output.reshape(output.shape[0], 4,256, 256)
            visualize_segmentation(output, batch_index=0, filename="./output/pred")
            visualize_segmentation(label, batch_index=0, filename="./output/label")
            visual_result(image[0],"./output/original")
            accuarcy=CaculateAcc(output,label)
            sum_accuarcy=sum_accuarcy+ accuarcy
            validation_iterator.set_description('ValAcc= %3.3f %%' % (sum_accuarcy*100/(i+1)))
    
if __name__=="__main__":
    main()