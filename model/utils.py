import os
import sys
import torch
from PIL import Image
import numpy as np
import random
import tqdm
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import  Dataset
from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def PrintModelInfo(model):
    """Print the parameter size and shape of model detail"""
    total_params = 0
    for name, param in model.named_parameters():
        num_params = torch.prod(torch.tensor(param.shape)).item() * param.element_size() / (1024 * 1024)  # 转换为MB
        print(f"{name}: {num_params:.4f} MB, Shape: {param.shape}")
        total_params += num_params
    print(f"Total number of parameters: {total_params:.4f} MB")     

def min_max_normalize(image):
    np_image = np.array(image).astype(np.float32)
    np_image = (np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image))
    return torch.tensor(np_image)

def visual_result(input,filename):
    if len(input.shape)==4:
        np_image = input[0].cpu().permute(1,2,0).numpy()  # 将通道维度移到最后
    elif len(input.shape)==3:
        np_image = input.cpu().permute(1,2,0).numpy()  # 将通道维度移到最后
    if np_image.min()<0:    
        np_image = np_image * 0.5 + 0.5  # 假设图像已归一化为[-1, 1]
    plt.imshow(np_image)
    plt.axis('off')
    plt.savefig(filename)  # 在绘制图像后保存  
    
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  
    ])
    image=transform(image)
    #visual_result(image,"out_original.jpg")
    return image

def preprocess_image_gray(image_path):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image=transform(image)
    #visual_result(image,"out_mask.jpg")
    def convert_to_labels(image_tensor, thresholds):
        """
        将灰度图像张量转换为标签。
        
        :param image_tensor: 预处理后的图像张量，形状为 [1, H, W]
        :param thresholds: 标签映射字典，键是标签值，值是对应的灰度阈值
        :return: 标签图像，形状为 [H, W]
        """
        # 将图像张量转换为 NumPy 数组
        image_np = image_tensor.squeeze().cpu().numpy()  # 去除通道维度，得到 [H, W]
        image_np.max()
        image_np.min()
        # 初始化标签图像
        h, w = image_np.shape
        labels = np.zeros((h, w), dtype=np.uint8)
        
        # 将灰度值转换为标签
        for label, threshold in thresholds.items():
            labels[image_np >= threshold] = label
        
        return labels
    
    thresholds = {
        0: 0,    # 背景（阈值 0）
        1: 0.148,   # 小麦（阈值 38）
        2: 0.292,   # 芝麻（阈值 75）
        3: 0.441    # 玉米（阈值 113）
    }
    label=convert_to_labels(image,thresholds)
    def show_label(label):
        plt.figure(figsize=(10, 10))
        plt.imshow(label, cmap='tab20b', interpolation='nearest')  # 使用 'tab20b' 色图，适合多个标签
        plt.colorbar()  # 添加颜色条
        plt.title('Label Visualization')
        plt.savefig("label.jpg")  # 保存图片到指定路径
        plt.close()  # 关闭图形窗口以释放内存
    image = image.squeeze(0)  # [256, 256]
    num_classes = len(thresholds)
    label_one_hot = torch.zeros(( num_classes,256, 256), dtype=torch.float32)
    label = torch.from_numpy(label)  # 将 numpy 数组转换为 Tensor
    for class_idx in range(num_classes):
        label_one_hot[class_idx,:, :] = (label == class_idx).float()  
    return label_one_hot

def CaculateAcc(predictions, labels):
    """
    计算语义分割的精确度

    参数:
    - predictions (Tensor): 模型输出，形状为 [batch_size, num_classes, height, width]
    - labels (Tensor): 真实标签，形状为 [batch_size, num_classes, height, width] (one-hot 编码)

    返回:
    - accuracy (float): 精确度，取值范围 [0, 1]
    """
    # 确保 predictions 和 labels 都在相同的设备上
    device = predictions.device
    labels = labels.to(device)
    if predictions.dim() != 4:
        raise ValueError(f"predictions 的维度应该是 4，但实际是 {predictions.dim()}")
    if labels.dim() != 4:
        raise ValueError(f"labels 的维度应该是 4，但实际是 {labels.dim()}")
    labels = torch.argmax(labels, dim=1)  # labels 的新形状为 [batch_size, height, width]
    _, predicted = torch.max(predictions, 1)  # predicted 的形状为 [batch_size, height, width]
    correct_pixels = (predicted == labels).sum().item()
    total_pixels = labels.numel()
    accuracy = correct_pixels / total_pixels
    
    return accuracy

def CalculateMiou(predictions, labels, num_classes):
    device = predictions.device
    labels = labels.to(device)
    if predictions.dim() != 4:
        raise ValueError(f"predictions 的维度应该是 4，但实际是 {predictions.dim()}")
    if labels.dim() != 4:
        raise ValueError(f"labels 的维度应该是 4，但实际是 {labels.dim()}")
    _, predicted = torch.max(predictions, dim=1)  # predicted 的形状为 [batch_size, height, width]
    labels = torch.argmax(labels, dim=1)  # labels 的形状为 [batch_size, height, width]
    ious = []
    for cls in range(num_classes):
        intersection = ((predicted == cls) & (labels == cls)).sum().item()
        union = ((predicted == cls) | (labels == cls)).sum().item()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union

        ious.append(iou)
    ious = [iou for iou in ious if not torch.isnan(torch.tensor(iou))]
    miou = sum(ious) / len(ious) if ious else float('nan')

    return miou
   
def load_and_cache_withlabel(data_path,label_path,cache_file,shuffle=False):
    if cache_file is not None and os.path.exists(cache_file):
        print("Loading features from cached file ", cache_file)
        features = torch.load(cache_file)
    else:
        print("Creating features from dataset at ", data_path)
        def processdata(data_path,label_path):
            images,labels=[],[]
            img_names = os.listdir(data_path)
            img_names.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            label_names = os.listdir(label_path)
            label_names.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            for img_name,label_name in zip(img_names,label_names):
                img_path = os.path.join(data_path, img_name)
                lab_path = os.path.join(label_path, label_name)
                images.append(img_path)
                labels.append(lab_path)
            return images,labels
        images,labels=processdata(data_path,label_path)
        features=[]
        total_iterations = len(images) 
        for image,label in tqdm.tqdm(zip(images,labels),total=total_iterations):
            processed_image=preprocess_image(image)
            label=preprocess_image_gray(label)
            feature={
                "images":processed_image,
                "label":label
            }
            features.append(feature)
 
        if shuffle:
            random.shuffle(features)
        if not os.path.exists(cache_file):
            print("Saving features into cached file ", cache_file)
            torch.save(features, cache_file)
    return features

class StrawDataset(Dataset):
    def __init__(self,features,num_instances):
        self.feature=features
        self.num_instances=num_instances
    
    def __len__(self):
        return int(self.num_instances)
    
    def __getitem__(self, index):
        feature = self.feature[index]
        image=feature["images"]
        label=feature["label"]
        return image,label
    
