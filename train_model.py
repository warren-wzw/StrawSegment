import os
from re import U
import sys
import torch
import tqdm 
import torch.nn as nn

os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model.unet import UNet,UNet_Se,UNet_Atten,UNet_Se_Atten,UNet_Se_Atten_Trans,UNet_Atten_Trans
from model.unet import UNet2,UNet2_se,UNet2_Atten,UNet2_se_Atten,UNet2_se_Atten_Trans,UNet2_Atten_Trans
from model.SegNet import SegNet
from model.DDRNet import DualResNet,BasicBlock
from torch.utils.data import (DataLoader)
from datetime import datetime
from model.utils import StrawDataset,load_and_cache_withlabel,get_linear_schedule_with_warmup,PrintModelInfo,CaculateAcc,CalculateMiou
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
    
BATCH_SIZE=25
EPOCH=100
LR=1e-5
TENSORBOARDSTEP=500
SAVE_MODEL='./output/output_model/'
MODEL_NAME="UNet2_se_Atten_Trans.pth"
PRETRAINED_MODEL_PATH=f" "
PRETRAINED_MODEL_PATH=f"./output/output_model/UNet2_se_Atten.pth"
Pretrain=False if PRETRAINED_MODEL_PATH ==" " else True
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
TF_ENABLE_ONEDNN_OPTS=0

"""dataset"""
train_type="train"
data_path_train=f"./dataset/src/{train_type}"
label_path_train=f"./dataset/label/{train_type}"
cached_file=f"./dataset/cache/{train_type}.pt"
val_type="val"
data_path_val=f"./dataset/src/{val_type}"
label_path_val=f"./dataset/label/{val_type}"
cached_file_val=f"./dataset/cache/{val_type}.pt"

def CreateDataloader(image_path,label_path,cached_file):
    features = load_and_cache_withlabel(image_path,label_path,cached_file,shuffle=True)  
    num_features = len(features)
    num_train = int(1* num_features)
    train_features = features[:num_train]
    dataset = StrawDataset(features=train_features,num_instances=num_train)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def main():
    global_step=0
    """Define Model"""
    model=UNet2_se_Atten_Trans(3,4).to(DEVICE)
    #model=SegNet().to(DEVICE)
    #model=UNet2_Atten_Trans().to(DEVICE)
    #model=UNet2_se(BasicBlock, [2, 2, 2, 2], num_classes=4, planes=32, spp_planes=128, head_planes=64, augment=False).to(DEVICE)
    PrintModelInfo(model)
    """Pretrain"""
    if Pretrain:
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH),strict=False)
    """Create dataloader"""
    dataloader_train=CreateDataloader(data_path_train,label_path_train,cached_file)
    dataloader_val=CreateDataloader(data_path_val,label_path_val,cached_file_val)
    total_steps = len(dataloader_train) * EPOCH
    """Loss function"""
    criterion = nn.CrossEntropyLoss()
    """Optimizer"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    """ Train! """
    best_accuarcy=0 
    model.train()
    torch.cuda.empty_cache()
    scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * total_steps , total_steps)
    tb_writer = SummaryWriter(log_dir='./output/tflog/') 
    print("  ************************ Running training ***********************")
    print("  Num Epochs = ", EPOCH)
    print("  Batch size per node = ", BATCH_SIZE)
    print("  Num examples = ", dataloader_train.sampler.data_source.num_instances)
    print(f"  Pretrained Model is ")
    print(f"  Save Model as {SAVE_MODEL+MODEL_NAME}")
    print("  ****************************************************************")
    start_time=datetime.now()
    for epoch_index in range(EPOCH):
        loss_sum=0
        sum_test_accuarcy=0
        train_iterator = tqdm.tqdm(dataloader_train, initial=0,desc="Iter", disable=False)
        for step, (image,label) in enumerate(train_iterator):
            image,label= image.to(DEVICE),label.to(DEVICE)
            optimizer.zero_grad()
            output=model(image)
            output = output.reshape(output.shape[0],4,256, 256)
            accuarcy=CaculateAcc(output,label)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            loss_sum=loss_sum+loss.item()
            sum_test_accuarcy=sum_test_accuarcy+accuarcy
            current_lr= scheduler.get_last_lr()[0]
            """ tensorbooard """
            if  global_step % TENSORBOARDSTEP== 0 and tb_writer is not None:
                tb_writer.add_scalar('train/lr', current_lr, global_step=global_step)
                tb_writer.add_scalar('train/loss', loss.item(), global_step=global_step)
            global_step=global_step+1
            scheduler.step()
            train_iterator.set_description('Epoch=%d, Acc= %3.3f %%,loss=%.6f, lr=%9.7f' 
                                           % (epoch_index,(sum_test_accuarcy/(step+1))*100, loss_sum/(step+1), current_lr))
        """ validation """
        sum_accuarcy=0
        sum_iou=0
        model.eval()
        with torch.no_grad():
            validation_iterator = tqdm.tqdm(dataloader_val, initial=0,desc="Iter", disable=False)
            for i,  (image,label) in enumerate(validation_iterator):
                image,label= image.to(DEVICE),label.to(DEVICE) 
                output=model(image)
                output = output.reshape(output.shape[0],4,256,256)
                accuarcy=CaculateAcc(output,label)
                iou=CalculateMiou(output,label,output.shape[1])
                sum_accuarcy=sum_accuarcy+ accuarcy
                sum_iou=sum_iou+iou
                validation_iterator.set_description('ValAcc= %3.3f %%,miou= %3.3f ' % (sum_accuarcy*100/(i+1),sum_iou/(i+1)))
        
        if sum_accuarcy/(i+1) > best_accuarcy:
            best_accuarcy = sum_accuarcy/(i+1)
            if not os.path.exists(SAVE_MODEL):
                os.makedirs(SAVE_MODEL)
            torch.save(model.state_dict(), os.path.join(SAVE_MODEL,f"{MODEL_NAME}"))
            print("->Saving model {} at {}".format(SAVE_MODEL+f"{MODEL_NAME}", 
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    end_time=datetime.now()
    print("Training consume :",(end_time-start_time)/60,"minutes")
    
if __name__=="__main__":
    main()