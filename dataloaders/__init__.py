from dataloaders.datasets import cityscapes, kd, coco, combine_dbs, pascal, sbd, nightcity,video_data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0,1'
def make_data_loader(args, **kwargs):
    if args.dist:
        print("=> Using Distribued Sampler")
        if args.dataset == 'cityscapes':
            if args.autodeeplab == 'search':
                train_set1, train_set2 = cityscapes.twoTrainSeg(args)
                num_class = train_set1.NUM_CLASSES
                sampler1 = torch.utils.data.distributed.DistributedSampler(train_set1)
                sampler2 = torch.utils.data.distributed.DistributedSampler(train_set2)
                train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=False, sampler=sampler1, **kwargs)
                train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, shuffle=False, sampler=sampler2, **kwargs)

            elif args.autodeeplab == 'train':
                train_set = cityscapes.CityscapesSegmentation(args, split='retrain')
                num_class = train_set.NUM_CLASSES
                sampler1 = torch.utils.data.distributed.DistributedSampler(train_set)
                train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=sampler1, **kwargs)

            else:
                raise Exception('autodeeplab param not set properly')

            val_set = cityscapes.CityscapesSegmentation(args, split='val')
            test_set = cityscapes.CityscapesSegmentation(args, split='test')
            sampler3 = torch.utils.data.distributed.DistributedSampler(val_set)
            sampler4 = torch.utils.data.distributed.DistributedSampler(test_set)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, sampler=sampler3, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, sampler=sampler4, **kwargs)

            if args.autodeeplab == 'search':
                return train_loader1, train_loader2, val_loader, test_loader, num_class
            elif args.autodeeplab == 'train':
                return train_loader,val_loader,test_loader, num_class
        else:
            raise NotImplementedError


        ################################################# Night City #################################################
    elif args.dataset == 'nightcity':
        if args.autodeeplab == 'search':
            train_set1, train_set2 = nightcity.twoTrainSeg(args)
            num_class = train_set1.NUM_CLASSES
            train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=True, **kwargs)
            train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, shuffle=True, **kwargs)
        elif args.autodeeplab == 'train':
            train_set = nightcity.NightCitySegmentation(args, split='retrain')
            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            # train_set = nightcity_new.NightSegmentation(args,split='train')
            # num_class = train_set.NUM_CLASS
            # train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True,**kwargs)
        else:
            raise Exception('autodeeplab param not set properly')

        val_set = nightcity.NightCitySegmentation(args, split='val')
        test_set = nightcity.NightCitySegmentation(args, split='test') # NightCity doesn't have test set
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        # val_set = nightcity_new.NightSegmentation(args,split='val')
        # test_set = nightcity_new.NightSegmentation(args,split='test')
        # val_loader = DataLoader(val_set,batch_size=args.batch_size, shuffle=False,**kwargs)
        # test_loader = DataLoader(test_set,batch_size=args.batch_size,shuffle=False,**kwargs)

        if args.autodeeplab == 'search':
            return train_loader1, train_loader2, val_loader, test_loader, num_class
        elif args.autodeeplab == 'train':
            return train_loader,val_loader,test_loader, num_class
        ################################################# cityscapes  #################################################

        elif args.dataset == 'cityscapes':
            if args.autodeeplab == 'search':
                train_set1, train_set2 = cityscapes.twoTrainSeg(args)
                num_class = train_set1.NUM_CLASSES
                train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=True, **kwargs)
                train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, shuffle=True, **kwargs)
            elif args.autodeeplab == 'train':
                train_set = cityscapes.CityscapesSegmentation(args, split='retrain')
                num_class = train_set.NUM_CLASSES
                train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            else:
                raise Exception('autodeeplab param not set properly')

            val_set = cityscapes.CityscapesSegmentation(args, split='val')
            test_set = cityscapes.CityscapesSegmentation(args, split='test')
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

            if args.autodeeplab == 'search':
                return train_loader1, train_loader2, val_loader, test_loader, num_class
            elif args.autodeeplab == 'train':
                return train_loader,val_loader,test_loader, num_class



        else:
            raise NotImplementedError
