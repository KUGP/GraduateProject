import os
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from mypath import Path     # 파일 없음 구현 X
from dataloaders import make_data_loader
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
from torchvision.utils import make_grid


print('working with pytorch version {}'.format(torch.__version__))
print('with cuda version {}'.format(torch.version.cuda))
print('cudnn enabled: {}'.format(torch.backends.cudnn.enabled))
print('cudnn version: {}'.format(torch.backends.cudnn.version()))




torch.backends.cudnn.benchmark = True

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # self.opt_level = args.opt_level

        # Define Dataloader depends on args.dataset
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last':True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        args.num_classes = self.nclass
        self.opt_level = args.opt_level


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

        checkpoint_name = os.path.join('run',args.dataset,'NC_train5_epoch491.pth')
        print('Checkpoint name : {}'.format(checkpoint_name))
        checkpoint = torch.load(checkpoint_name)
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
        # model.module.load_state_dict(state_dict)
        model.load_state_dict(state_dict)
        #
        # Resuming checkpoint
        # self.best_pred = 0.0
        # if args.resume is not None:
        #     if not os.path.isfile(args.resume):
        #         raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        #     checkpoint = torch.load(args.resume)
        #     args.start_epoch = checkpoint['epoch']
        #
        #     # if the weights are wrapped in module object we have to clean it
        #     if args.clean_module:
        #         self.model.load_state_dict(checkpoint['state_dict'])
        #         state_dict = checkpoint['state_dict']
        #         new_state_dict = OrderedDict()
        #         for k, v in state_dict.items():
        #             name = k[7:]  # remove 'module.' of dataparallel
        #             new_state_dict[name] = v
        #         # self.model.load_state_dict(new_state_dict)
        #         copy_state_dict(self.model.state_dict(), new_state_dict)
        #
        #     else:
        #         if torch.cuda.device_count() > 1 or args.load_parallel:
        #             # self.model.module.load_state_dict(checkpoint['state_dict'])
        #             # copy_state_dict(self.model.module.state_dict(), checkpoint['state_dict'])
        #             copy_state_dict(self.model.state_dict(), checkpoint['state_dict'])
        #
        #         else:
        #             # self.model.load_state_dict(checkpoint['state_dict'])
        #             copy_state_dict(self.model.state_dict(), checkpoint['state_dict'])
        #
        #
        #     # if not args.ft:
        #         # self.optimizer.load_state_dict(checkpoint['optimizer'])
        #         # copy_state_dict(self.optimizer.state_dict(), checkpoint['optimizer'])
        #
        #     # self.best_pred = checkpoint['best_pred']
        #     print("=> loaded checkpoint '{}' (epoch {})"
        #           .format(args.resume, checkpoint['epoch']))


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
        val_loss = 0.0
        val_Acc = 0.0
        self.union_meter.reset()
        self.intersection_meter.reset()
        self.evaluator.reset()

        for i, sample in enumerate(self.test_loader): #if you want to validate with nighttimedriving change val_loader -> test_loader
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            val_loss += loss.item()

            #convert into numpy
            pred = output.data.cpu().numpy()
            targets = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            # Add batch sample into evaluator
            self.evaluator.add_batch(targets, pred)
            self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, i)

            # Fast test during the training
            Acc = self.evaluator.Pixel_Accuracy()
            intersection, union, targets = self.intersectionAndUnion(pred, targets, 19)

            val_Acc += Acc
            self.union_meter.update(union)
            self.intersection_meter.update(intersection)
            print('EPOCH {}\tITER {}/{}'.format(epoch,i,len(self.val_loader)))

        #
        epoch_loss = val_loss/len(self.val_loader)
        epoch_acc = val_Acc/len(self.val_loader)
        iou_class = self.intersection_meter.sum / (self.union_meter.sum + 1e-10)
        epoch_miou = np.mean(iou_class)

        self.writer.add_scalar('val/total_loss_epoch', epoch_loss, epoch)
        self.writer.add_scalar('val/Acc', epoch_acc, epoch)
        self.writer.add_scalar('val/mIoU', epoch_miou, epoch)


        print("EPOCH : {}\tLOSS : {}\tACC : {}\tmIoU: {}".format(epoch, epoch_loss,
                                                      epoch_acc,
                                                     epoch_miou))

        # Show 10 * 3 inference results each epoch
        # self.summary.visualize_image(self.writer, self.args.dataset, image, target, output['out'], epoch+1)



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
            'nightcity':150
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
    for epoch in range(10): # validation only
        trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()