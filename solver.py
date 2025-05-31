import os
import numpy as np
import time
from datetime import datetime
from PIL import Image

import torch

from FAT_Net import FAT_Net
from functions import  cross_entropy_loss_RCF
import torchvision
from UNet import U_Net
from SegNet import SegNet
from UNet_VGG import UNet_VGG
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from swin_transformer import SwinTransformer
from network import U_Net
import cv2
import segmentation_models_pytor as smp
import csv
from misc import *
import os
import argparse
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from tensorboardX import SummaryWriter
from network import R2U_Net,AttU_Net,R2AttU_Net
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('mylogdir')


class Solver(object):
    def __init__(self, config, model, train_loader, valid_loader, test_loader):
        
		# Data loader
        self.mode = config.mode
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

		# Hyper-parameters
        self.lr = config.lr
        self.optimizer= config.optimizer
        self.beta1 = config.beta1
        self.beta2 = config.beta2

		# Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.num_epochs_decay = config.num_epochs_decay
        
		# Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.SR_path = config.SR_path
        self.model_type = model
        self.dataset = config.dataset
        self.loss = config.loss_type

        # Report file

        self.report_file = config.report_file

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch

        self.augmentation_prob = config.augmentation_prob
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion1 = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.criterion2 = mIoULoss(threshold=config.loss_threshold).to(self.device)
        self.criterion3 = DiceLoss(threshold=config.loss_threshold).to(self.device)
        #self.criterionSeg = BinaryFocalLoss().to(self.device)







    def build_model(self):
        """Build generator and discriminator."""
        print("initialize training...")
        if self.model_type == 'FAT_Net':
            self.unet = FAT_Net()

        else:
           self.unet = U_Net(self.img_ch,self.output_ch)


        if self.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.unet.parameters(),self.lr, [self.beta1, self.beta2], weight_decay=1e-4)
        else:
            self.optimizer = optim.SGD(self.unet.parameters(), lr=self.lr, momentum=self.beta1, weight_decay=2e-4)
        #self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8, last_epoch=-1)

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, patience=self.num_epochs_decay, verbose=True)
        
        self.unet.to(self.device)

        self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        self.report.write('\n'+str(model))
        print(name)
        self.report.write('\n'+str(name))
        print("The number of parameters: {}".format(num_params))
        self.report.write("\n The number of parameters: {}".format(num_params))
        
    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()
        

    def train(self,loss):
        t = time.time()
        self.loss = loss
        isExist = os.path.exists(self.result_path + self.model_type+ '_' + loss)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.result_path + self.model_type + '_' + loss)
        self.result_path_loss = self.result_path + self.model_type + '_' + loss + '/'
        self.report = open(
            self.result_path_loss+ self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '.txt',
            'a+')
        self.report.write('\n' + str(datetime.now()))
        # self.report.write('\n' + str(config))

        self.f1 = open(os.path.join(self.result_path_loss,
                                    self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '_train.csv'),
                       'a', encoding='utf-8', newline='')
        self.f2 = open(os.path.join(self.result_path_loss,
                                    self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '_val.csv'),
                       'a', encoding='utf-8', newline='')
        self.model_save_path = os.path.join(self.model_path,
                                            self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '.pkl')
        self.model_save_path1 = os.path.join(self.model_path,
                                            self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss)

        self.build_model()
        wr1 = csv.writer(self.f1)
        wr1.writerow(
            ['Epoch', 'Acc', 'RC', 'SP', 'PC', 'F1', 'IoU', 'mIoU', 'DC',
             'LR', 'loss'])
        wr2 = csv.writer(self.f2)
        wr2.writerow(
            ['Epoch', 'Acc', 'RC', 'SP', 'PC', 'F1', 'IoU', 'mIoU', 'DC',
             'LR', 'loss'])

        # U-Net Train
        if os.path.isfile(self.model_save_path):
			# Load the pretrained Encoder
            self.unet = torch.load(self.model_save_path)
            print('%s is Successfully Loaded from %s'%(self.model_type,self.model_save_path))
            self.report.write('\n %s is Successfully Loaded from %s'%(self.model_type,self.model_save_path))
        else:
            # Train for Encoder
            best_unet_score = 0.
            results = [["Loss",[],[]],["Acc",[],[]],["RC",[],[]],["SP",[],[]],["PC",[],[]],["F1",[],[]],["IoU",[],[]],["mIoU",[],[]],["DC",[],[]]]

            # for epoch1 in range(self.num_epochs):
            #     print("77777777777777777777")
            #     print(epoch1)

            for epoch in range(self.num_epochs):
                # print("88888888888888888888")
                # print(epoch)
                self.unet.train(True)
                train_loss = 0.
				
                acc = 0.	# Accuracy
                RC = 0.		# Sensitivity (Recall)
                SP = 0.		# Specificity
                PC = 0. 	# Precision
                F1 = 0.		# F1 Score
                IoU = 0     # Intersection over Union (Jaccard Index)
                mIoU = 0.	# mean of Intersection over Union (mIoU)
                DC = 0.		# Dice Coefficient
                length = 0
                buff = []
                
                for i, (image, GT, name) in enumerate(self.train_loader):
                    # print('image')
                    # print(i)
                    # SR : Segmentation Result
                    # GT : Ground Truth
                    image = image.to(self.device)
                    GT = GT.to(self.device)
# ----------------------------------UNet--------------------------------------------------------------

                    SR = self.unet(image)
                    SR = SR.view(-1)
                    GT = GT.view(-1)

                    loss1 = self.criterion1(SR, GT)
                    loss2 = self.criterion2(SR, GT)
                    loss3 = self.criterion3(SR,GT)

                    
                    total_loss = loss1 + (factor*(loss2+loss3))
                    

                    # Backprop + optimize
                    self.reset_grad()
                    total_loss.backward()
                    self.optimizer.step()
                    
                    # Detach from GPU memory
                    SR = SR.detach()
                    GT = GT.detach()
                    
                    train_loss += total_loss.detach().item()
                    acc += get_accuracy(SR,GT)
                    RC += get_Recall(SR,GT)
                    SP += get_specificity(SR,GT)
                    PC += get_Precision(SR,GT)
                    F1 += get_F1(SR,GT)
                    buff = get_mIoU(SR,GT)
                    IoU += buff[0]
                    mIoU += buff[1]
                    DC += get_DC(SR,GT)
                    length += 1

                train_loss = train_loss/length
                acc = acc/length
                RC = RC/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                IoU = IoU/length
                mIoU = mIoU/length
                DC = DC/length
                
                results[0][1].append((train_loss))
                results[1][1].append((acc*100))
                results[2][1].append((RC*100))
                results[3][1].append((SP*100))
                results[4][1].append((PC*100))
                results[5][1].append((F1*100))
                results[6][1].append((IoU*100))
                results[7][1].append((mIoU*100))
                results[8][1].append((DC*100))
                
                # Print the log info
                print('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f' % (
                    epoch+1,self.num_epochs,train_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                self.report.write('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f' % (
                    epoch+1,self.num_epochs,train_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                wr1.writerow(
                    [epoch + 1, acc, RC, SP, PC, F1, IoU, mIoU, DC, self.lr, train_loss])
                writer.add_scalar("Loss/train", train_loss, epoch+1)
                writer.add_scalar("Precision/train", PC, epoch + 1)
                writer.add_scalar("Recall/train", RC, epoch + 1)
                writer.add_scalar("F1 Score/train", F1, epoch + 1)
                writer.add_scalar("mIoU/train", mIoU, epoch + 1)


				# Clear unoccupied GPU memory after each epoch
                torch.cuda.empty_cache()
                
#===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()
                valid_loss = 0.

                acc = 0.	# Accuracy
                RC = 0.		# Sensitivity (Recall)
                SP = 0.		# Specificity
                PC = 0. 	# Precision
                F1 = 0.		# F1 Score
                IoU = 0     # Intersection over Union (Jaccard Index)
                mIoU = 0.	# mean of Intersection over Union (mIoU)
                DC = 0.		# Dice Coefficient
                length=0
                buff = []
                
                for i, (image, GT, name) in enumerate(self.valid_loader):
                    
                    # SR : Segmentation Result
                    # GT : Ground Truth
                    image = image.to(self.device)
                    GT = GT.to(self.device)
                    GT_f = GT

#-------------------------------------UNet-------------------------------------------------------
                    SR = self.unet(image)
                    SR_f = SR.view(-1)
                    GT_f = GT.view(-1)
                    loss_val_1 = self.criterion1(SR_f, GT_f)
                    loss_val_2 = self.criterion2(SR_f, GT_f)
                    loss_val_3 = self.criterion3(SR_f,GT_f)

                    total_loss = loss_val_1 + (factor*(loss_val_2+loss_val_3))

                    
                    # Detach from GPU memory
                    SR_f = SR_f.detach()
                    GT_f = GT_f.detach()
                    
                    # Get metrices results
                    valid_loss += total_loss.detach().item()
                    acc += get_accuracy(SR_f,GT_f)
                    RC += get_Recall(SR_f,GT_f)
                    SP += get_specificity(SR_f,GT_f)
                    PC += get_Precision(SR_f,GT_f)
                    F1 += get_F1(SR_f,GT_f)
                    buff = get_mIoU(SR_f,GT_f)
                    IoU += buff[0]
                    mIoU += buff[1]
                    DC += get_DC(SR_f,GT_f)
                    length += 1
                    
                valid_loss = valid_loss/length
                acc = acc/length
                RC = RC/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                IoU = IoU/length
                mIoU = mIoU/length
                DC = DC/length
                unet_score = mIoU
                
                results[0][2].append((valid_loss))
                results[1][2].append((acc*100))
                results[2][2].append((RC*100))
                results[3][2].append((SP*100))
                results[4][2].append((PC*100))
                results[5][2].append((F1*100))
                results[6][2].append((IoU*100))
                results[7][2].append((mIoU*100))
                results[8][2].append((DC*100))

                print('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f'%(
                    valid_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                self.report.write('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f'%(
                    valid_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))

                wr2.writerow([epoch+1 ,acc,RC,SP,PC,F1,IoU,mIoU,DC,self.lr,valid_loss])
                writer.add_scalar("Loss/val", valid_loss, epoch + 1)
                writer.add_scalar("Precision/val", PC, epoch + 1)
                writer.add_scalar("Recall/val", RC, epoch + 1)
                writer.add_scalar("F1 Score/val", F1, epoch + 1)
                writer.add_scalar("mIoU/val", mIoU, epoch + 1)

                
                # Decay learning rate
                self.lr_scheduler.step(valid_loss)
                    
                # Save Best U-Net model
                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    print('\nBest %s model score : %.4f'%(self.model_type,best_unet_score))
                    self.report.write('\nBest %s model score : %.4f'%(self.model_type,best_unet_score))
                    torch.save(self.unet,self.model_save_path)
                epoch_custom = epoch + 1
                if epoch_custom % 10 ==0:
                    torch.save(self.unet, self.model_save_path1+'_'+str(epoch_custom)+'.pkl')
                    
                    
                if unet_score > 0.9:
                    torchvision.utils.save_image(image.data.cpu(),os.path.join(
                        self.result_path_loss,self.report_file+'_%s_valid_%d_image.png'%(self.model_type,epoch+1)))
                    torchvision.utils.save_image(SR.data.cpu(),os.path.join(
                        self.result_path_loss,self.report_file+'_%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
                    torchvision.utils.save_image(GT.data.cpu(),os.path.join(
                        self.result_path_loss,self.report_file+'_%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
                
                # Clear unoccupied GPU memory after each epoch
                torch.cuda.empty_cache()
            
            displayfigures(results, self.result_path_loss, self.report_file, self.dataset, self.model_type)
        
        elapsed = time.time() - t
        print("\nElapsed time: %f seconds.\n\n" %elapsed)
        self.report.write("\nElapsed time: %f seconds.\n\n" %elapsed)
        self.report.close()
        self.f1.close()
        self.f2.close()
        writer.close()

    def get_gradCAM(self,image,SR, GT,size):

        total_loss = self.criterion1(SR, GT)
        total_loss.backward()
        gradients = self.unet.get_activation_gradients()
        pooled_gradients = torch.mean(gradients, dim=[0,2,3])
        activations = self.unet.get_activations(image).detach()
        for i in range(activations.shape[1]):
            activations[:,i,:,:] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim = 1).squeeze().cpu()
        heatmap = nn.ReLU()(heatmap)
        heatmap /= torch.max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        image = image.squeeze(0)
        image = image.permute(1,2,0)
        image = image.cpu().numpy()
        image = np.uint8(image * 255)
        # image = Image.fromarray(image)
        # heatmap = Image.fromarray(heatmap).convert("L")
        heatmap = cv2.resize(heatmap, (320, 320))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # heatmap = heatmap.resize((320, 320), Image.NEAREST)
        overlay = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)
        return overlay, heatmap
                    
    def test(self,loss, data, model):
		#===================================== Test ====================================#
        
        # Load Trained U-Net
        if os.path.isfile(self.model_path+'Skin_Lesion_Segmentation_'+data+'_'+model+'_'+loss+'.pkl'):
			# Load the pretrained Encoder
            self.unet = torch.load(self.model_path+'Skin_Lesion_Segmentation_'+data+'_'+model+'_'+loss+'.pkl')
            print('%s is Successfully Loaded from %s'%(self.model_type,self.model_path))
            # self.report.write('\n%s is Successfully Loaded from %s'%(self.model_type,self.model_path))
        else: 
            print("Trained model NOT found, Please train a model first")
            # self.report.write("\nTrained model NOT found, Please train a model first")
            return
        isExist = os.path.exists(self.SR_path+self.model_type + '_' + loss)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.SR_path + self.model_type + '_' + loss)
        self.model_path_loss = self.SR_path + self.model_type + '_' + loss + '/'
        self.test_acc = open(
            self.model_path_loss + self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '_test.csv',
            'a+')

        wr_test = csv.writer(self.test_acc)
        wr_test.writerow(
            ['Recall', 'Precision', 'F1', 'mIoU'])

        self.unet.train(False)
        self.unet.eval()
        
        acc = 0.	# Accuracy
        RC = 0.		# Sensitivity (Recall)
        SP = 0.		# Specificity
        PC = 0. 	# Precision
        F1 = 0.		# F1 Score
        IoU = 0     # Intersection over Union (Jaccard Index)
        mIoU = 0.	# mean of Intersection over Union (mIoU)
        DC = 0.		# Dice Coefficient
        elapsed = 0.# Time of inference
        threshold = 0
        length = 0
        buff = []
        
        for i, (image, GT, name) in enumerate(self.test_loader):
            
            # SR : Segmentation Result
            # GT : Ground Truth
            image = image.to(self.device)
            GT = GT.to(self.device)
            
            #Time of inference
            t = time.time()

            SR = self.unet(image)


            elapsed = (time.time() - t)

            # heatmap, heat = self.get_gradCAM(image, SR, GT, size=320)
            # Detach from GPU memory
            SR = SR.detach()
            GT = GT.detach()
            SR_f = SR.view(-1)
            GT_f = GT.view(-1)
            

            acc += get_accuracy(SR_f,GT_f)
            RC += get_Recall(SR_f,GT_f)
            SP += get_specificity(SR_f,GT_f)
            PC += get_Precision(SR_f,GT_f)
            F1 += get_F1(SR_f,GT_f)
            buff = get_mIoU(SR_f,GT_f)
            IoU += buff[0]
            mIoU += buff[1]
            threshold = 0.5#buff[2]
            DC += get_DC(SR_f,GT_f)
            length += 1

            SR[SR < threshold] = 0
            SR[SR > threshold] = 1
            im = Image.fromarray(SR[0,:,:].squeeze().cpu().numpy() * 255).convert('L')
            imo = im.resize((256, 256), resample=Image.BILINEAR)
            imo.save(self.model_path_loss+name[0])

            # cv2.imwrite('SR/CAM/HeatMap/' + name[0], heatmap)
            # cv2.imwrite('SR/CAM/Heat/' + name[0], heat)
            

        acc = acc/length
        RC = RC/length
        SP = SP/length
        PC = PC/length
        F1 = F1/length
        IoU = IoU/length
        mIoU = mIoU/length
        DC = DC/length
        elapsed = elapsed/(length*SR.size(0))
        unet_score = mIoU

        wr_test.writerow(
            [RC, PC, F1, mIoU])
        print('Results have been Saved')

        
        self.test_acc.close()


        
        
