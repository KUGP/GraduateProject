import os
import pdb
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
import torch.optim as optim

import dataloaders
from utils.utils import AverageMeter
from utils.loss import build_criterion
from utils.step_lr_scheduler import Iter_LR_Scheduler
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args
from utils.summaries import TensorboardSummary
from utils.saver import Saver
from utils.metrics import Evaluator


torch.backends.cudnn.benchmark = True
### re_train decoded architecture




def main():
    print('working with pytorch version {}'.format(torch.__version__))
    print('with cuda version {}'.format(torch.version.cuda))
    print('cudnn enabled: {}'.format(torch.backends.cudnn.enabled))
    print('cudnn version: {}'.format(torch.backends.cudnn.version()))

    warnings.filterwarnings('ignore')
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    args = obtain_retrain_autodeeplab_args()
    saver = Saver(args)
    # saver.save_experiment_config()

    summary = TensorboardSummary(saver.experiment_dir)
    writer = summary.create_summary()

    model_fname = 'run/{}/{}_epoch%d.pth'.format(args.dataset,args.checkname)
    if args.dataset == 'pascal':
        raise NotImplementedError
    elif args.dataset == 'cityscapes':
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        dataset_loader, num_classes = dataloaders.make_data_loader(args, **kwargs)
        args.num_classes = num_classes

    elif args.dataset == 'nightcity':
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        train_loader,val_loader,test_loader, num_classes = dataloaders.make_data_loader(args, **kwargs)
        args.num_classes = num_classes
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.backbone == 'autodeeplab':
        model = Retrain_Autodeeplab(args)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    if args.criterion == 'Ohem':
        args.thresh = 0.7
        args.crop_size = [args.crop_size, args.crop_size] if isinstance(args.crop_size, int) else args.crop_size
        args.n_min = int((args.batch_size / len(args.gpu) * args.crop_size[0] * args.crop_size[1]) // 16)
    criterion = build_criterion(args)

    evaluator = Evaluator(num_classes)
    # print("##########################Model########################")
    # print(model)
    # print("#######################################################")

    model = nn.DataParallel(model).cuda()
    model.train()
    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    optimizer = optim.SGD(model.module.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    max_iteration = len(train_loader) * args.epochs
    scheduler = Iter_LR_Scheduler(args, max_iteration, len(train_loader))
    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {0}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {0} (epoch {1})'.format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('=> no checkpoint found at {0}'.format(args.resume))


    best_mIoU = 0.0
    ###################### EPOCHS LOOP ##################
    for epoch in range(start_epoch, args.epochs):
        # EPOCH loss ...
        train_loss = 0.0
        train_Acc = 0.0
        train_mIoU = 0.0

        val_loss = 0.0
        val_Acc = 0.0
        val_mIoU = 0.0


        # num_img_tr = len(train_loader)
        losses = AverageMeter()
        evaluator.reset()

        ######### TARGET TRAIN LOOP ##########
        model.train()
        print("##################### TRAINING START #####################")

        for i, sample in enumerate(train_loader):

            # print("len dataset_loader",len(dataset_loader))   187 for nightcity with batch_size 16

            cur_iter = epoch * len(train_loader) + i
            scheduler(optimizer, cur_iter)
            inputs = sample['image'].cuda()
            target = sample['label'].cuda()
            outputs = model(inputs)
            loss = criterion(outputs, target.type(torch.long))
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                pdb.set_trace()
            losses.update(loss.item(), args.batch_size)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print(target.size())
            # print(outputs.size())
            pred = outputs.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            targets = target.cpu().numpy()

            evaluator.add_batch(targets, pred)
            Acc = evaluator.Pixel_Accuracy()
            mIoU = evaluator.Mean_Intersection_over_Union()
            train_loss += loss.item()
            train_Acc += Acc
            train_mIoU += mIoU
            print('EPOCH: {0}\titer: {1}/{2}\tlr: {3:.6f}\tLOSS: {4:.4f}\tACC: {5}\tmIoU: {6}'.format(epoch+1, i+1,len(train_loader),
                                                                                                    scheduler.get_lr(optimizer),
                                                                                                    loss.item(),
                                                                                                    Acc,mIoU))
            # summary.visualize_image(writer, args.dataset, inputs, target, outputs, i + 1)

        print('EPOCH: {0}\tAvg_LOSS: {1:.4f}\tAvg_ACC: {2}\tAvg_mIoU: {3}'.format(epoch + 1,train_loss/len(train_loader),
                                                                                    train_Acc/len(train_loader),
                                                                                    train_mIoU/len(train_loader)))

        writer.add_scalar('train/Avg loss_epoch', train_loss/len(train_loader), epoch+1)
        writer.add_scalar('train/Avg mIoU', train_mIoU/len(train_loader), epoch+1)
        writer.add_scalar('train/Avg Acc', train_Acc/len(train_loader), epoch+1)
        summary.visualize_image(writer, args.dataset, inputs, target, outputs, epoch+1)


        evaluator.reset()
        model.eval()
        print("##################### VALIDATION START #####################")
        ######### TARGET VAL LOOP ##########
        with torch.no_grad():
            for i, sample in enumerate(val_loader):
                # print("len dataset_loader",len(dataset_loader))   187 for nightcity with batch_size 16

                cur_iter = epoch * len(val_loader) + i
                scheduler(optimizer, cur_iter)
                inputs = sample['image'].cuda()
                target = sample['label'].cuda()
                outputs = model(inputs)
                loss = criterion(outputs, target.type(torch.long))

                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    pdb.set_trace()
                losses.update(loss.item(), args.batch_size)


                pred = outputs.data.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                targets = target.cpu().numpy()


                evaluator.add_batch(targets, pred)
                Acc = evaluator.Pixel_Accuracy()
                mIoU = evaluator.Mean_Intersection_over_Union()

                val_loss += loss.item()
                val_Acc += Acc
                val_mIoU += mIoU
                # # Show 10 * 3 inference results each epoch
                # if i % (num_img_tr // 10) == 0:
                #     global_step = i + num_img_tr * epoch
                #     summary.visualize_image(writer, args.dataset, inputs, target, outputs, global_step)
                print('EPOCH: {0}\titer: {1}/{2}\tlr: {3:.6f}\tLOSS: {4:.4f}\tACC: {5}\tmIoU: {6}'.format(epoch + 1, i + 1,
                                                                                                      len(val_loader),
                                                                                                      scheduler.get_lr(
                                                                                                          optimizer),
                                                                                                      loss.item(),
                                                                                                      Acc, mIoU))


                summary.visualize_image(writer, args.dataset, inputs, target, outputs, i)

        print('EPOCH: {0}\tAvg_LOSS: {1:.4f}\tAvg_ACC: {2}\tAvg_mIoU: {3}'.format(epoch + 1,
                                                                              val_loss / len(val_loader),
                                                                              val_Acc / len(val_loader),
                                                                              val_mIoU / len(val_loader)))

        writer.add_scalar('val/Avg loss_epoch', val_loss / len(val_loader), epoch + 1)
        writer.add_scalar('val/Avg mIoU', val_mIoU / len(val_loader), epoch + 1)
        writer.add_scalar('val/Avg Acc', val_Acc / len(val_loader), epoch + 1)

        if val_mIoU/len(val_loader) > best_mIoU:
            best_mIoU = val_mIoU/len(val_loader)
            # is_best = True
            # saver.save_checkpoint({
            #     'epoch': epoch + 1,
            #     'state_dict': model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            # }, is_best)
            print("########### NEW BEST MODEL FOUND ###########")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_fname % (epoch + 1))



        # summary.visualize_image(writer, args.dataset, inputs, target, outputs, epoch)




        print('reset local total loss!')




    writer.close()



if __name__ == "__main__":
    main()
