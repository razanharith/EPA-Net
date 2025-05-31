import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
	

def main(config):
    cudnn.benchmark = True

    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
        config.result_path = os.path.join(config.result_path,config.model_type)
    
    print(config)

    train_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0)
    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0)
    

    

    # Train and sample the images
    if config.mode == 'train':
        for model in ['EPA_Net']:
		solver = Solver(config,model, train_loader, valid_loader, test_loader)
            for loss in ['BCE_Dice_mIoU']:
                solver.train(loss=loss)



    elif config.mode == 'test':

        for model in ['EPA_Net']:
            solver = Solver(config, model, train_loader, valid_loader, test_loader)

            for loss in ['BCE_Dice_mIoU']:

                solver.test(loss=loss, data='ISIC', model=model)

                solver = Solver(config, model, train_loader, valid_loader, test_loader)


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # training hyper-parameters
    parser.add_argument('--lr', type=float, default=0.0095)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--loss_threshold', type=float, default=0.5)
    parser.add_argument('--loss_type', type=str, default='BCE_Dice_mIoU', help='[BCE,BCE_mIoU,BCE_mIoU,BCE_Dice_mIoU]')
    parser.add_argument('--optimizer', type=str, default='SGD', help='[Adam,SGD]')
    parser.add_argument('--beta1', type=float, default=0.86)       # momentum1 in Adam or SGD
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.5)

    # misc
    parser.add_argument('--report_file', type=str, default='Skin_Lesion_Segmentation')
    parser.add_argument('--mode', type=str, default='train', help='[train,test]')
    parser.add_argument('--dataset', type=str, default='ISIC', help='ISIC')
    # parser.add_argument('--model_type', type=str, default='deepCrack', help='[U_Net,TMUNet,TUNet,HED,deepCrack]')
    parser.add_argument('--model_path', type=str, default='/DATA/home/...')
    parser.add_argument('--result_path', type=str, default='/DATA/home/...')
    parser.add_argument('--train_path', type=str, default='/Datasets/ISIC2017/train/')
    parser.add_argument('--valid_path', type=str, default='/Datasets/ISIC2017/valid/')
    parser.add_argument('--test_path', type=str, default='/Datasets/ISIC2017/test/')
    parser.add_argument('--SR_path', type=str, default='/DATA/home/..')
    
    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
    
