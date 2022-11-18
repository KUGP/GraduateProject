import os
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from mypath import Path     # 파일 없음 구현 X
from dataloaders import make_data_loader
from dataloaders.datasets import video_data
from torch.utils.data import DataLoader

from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.base_seg_saver import Saver
from utils.base_seg_saver import AverageMeter

from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from auto_deeplab import AutoDeeplab
from config_utils.validate_args import obtain_search_args
from utils.copy_state_dict import copy_state_dict
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from dataloaders.dataloader_utils import decode_seg_map_sequence
from dataloaders.dataloader_utils import decode_segmap
from PIL import Image


from torchvision.utils import make_grid


print('working with pytorch version {}'.format(torch.__version__))
print('with cuda version {}'.format(torch.version.cuda))
print('cudnn enabled: {}'.format(torch.backends.cudnn.enabled))
print('cudnn version: {}'.format(torch.backends.cudnn.version()))

#SAVING (input images, Segmentation result) for DEMO video


torch.backends.cudnn.benchmark = True

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.video_root = 'C:/Users/oem/Desktop/video_data'


        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()


        # Define Dataloader depends on args.dataset
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last':True}
        self.opt_level = args.opt_level
        self.val_set = video_data.VideoSegmentation(args, split='val')
        self.val_loader = DataLoader(self.val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        self.nclass = self.val_set.NUM_CLASSES
        args.num_classes = self.nclass




        # Define Criterion (LOSS)
        weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        # Define Searched Model
        model = Retrain_Autodeeplab(args)
        self.model = model


        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        self.intersection_meter = AverageMeter()
        self.union_meter = AverageMeter()


        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()


        # Using data parallel
        if args.cuda and len(self.args.gpu_ids) >1:
            if self.opt_level == 'O2' or self.opt_level == 'O3':
                print('currently cannot run with nn.DataParallel and optimization level', self.opt_level)
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids).cuda()

            patch_replication_callback(self.model)
            print('training on multiple-GPUs')

        #LOAD CHECKPOINT
        checkpoint_name = os.path.join('run','nightcity','NC_train5_epoch491.pth')
        print('Checkpoint name : {}'.format(checkpoint_name))
        checkpoint = torch.load(checkpoint_name)
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
        # model.module.load_state_dict(state_dict)
        model.load_state_dict(state_dict)
        #


        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def intersectionAndUnion(self, output, target, K, ignore_index=255):
        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert (output.ndim in [1, 2, 3])
        assert output.shape == target.shape
        output = output.reshape(output.size).copy()
        target = target.reshape(target.size)
        output[np.where(target == ignore_index)[0]] = ignore_index
        intersection = output[np.where(output == target)[0]]
        area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
        area_output, _ = np.histogram(output, bins=np.arange(K + 1))
        area_target, _ = np.histogram(target, bins=np.arange(K + 1))
        area_union = area_output + area_target - area_intersection
        return area_intersection, area_union, area_target


    ###################################### Validation ########################################
    def validation(self, epoch):
        self.model.eval()
        num = 0


        for i, sample in enumerate(self.val_loader): #if you want to validate with nighttimedriving change val_loader -> test_loader
            image = sample['image']
            if self.args.cuda:
                image = image.cuda()
            with torch.no_grad():
                output = self.model(image)


            #convert into numpy
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            img = image.data.cpu().numpy()
            # print(np.shape(img))
            # print(np.shape(pred[0]))

            # print(np.shape(mask))
            for jj in range(self.args.batch_size):
                # tmp = np.array(gt[jj]).astype(np.uint8)
                # segmap = decode_segmap(tmp, dataset=args.dataset)
                segmap = decode_segmap(pred[jj], 'nightcity',False)
                img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
                img_tmp *= (0.229, 0.224, 0.225)
                img_tmp += (0.485, 0.456, 0.406)
                img_tmp *= 255.0
                img_tmp = img_tmp.astype(np.uint8)


                # plt.figure()
                # plt.axis('off')
                # plt.title('display')
                # plt.subplot(211)
                # plt.imshow(img_tmp)
                # plt.subplot(212)
                # plt.imshow(segmap)
                # plt.figure()
                img_path = self.video_root + '/video_image' + '/' + f'image_{str(num).zfill(4)}' + '.png'
                mask_path = self.video_root + '/video_mask' + '/' + f'mask_{str(num).zfill(4)}' + '.png'

                if num > 0: #Saving point
                    plt.imshow(img_tmp)
                    plt.axis('off')
                    plt.savefig(img_path, bbox_inches='tight')
                    plt.imshow(segmap)
                    plt.axis('off')
                    plt.savefig(mask_path,bbox_inches='tight')
                num+=1

            # Add batch sample into evaluator
            # # grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
            # #                                                dataset='nightcity'), 4, normalize=False, range=(0, 255))
            # grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
            #                                   dataset='nightcity'),nrow= 4, normalize=False, range=(0, 255))
            # plt.imshow(grid_image)
            # plt.show()

            print('EPOCH {}\tITER {}/{}'.format(epoch,i,len(self.val_loader)))





def main():
    args = obtain_search_args()
    args.cuda = torch.cuda.is_available()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    if torch.cuda.is_available():
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 40,
            'pascal': 50,
            'kd':10,
            'nightcity':150,
            'video':5
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    #args.lr = args.lr / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()