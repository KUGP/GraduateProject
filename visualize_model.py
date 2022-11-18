import warnings
import numpy as np

import torch
import torch.utils.data
import torch.backends.cudnn

import dataloaders
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args
import matplotlib.pyplot as plt
from dataloaders.dataloader_utils import decode_segmap
from torchvision.utils import make_grid
from dataloaders.dataloader_utils import decode_seg_map_sequence


warnings.filterwarnings('ignore')
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
args = obtain_retrain_autodeeplab_args()


if args.dataset == 'cityscapes':
    kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
    train_loader, val_loader, test_loader, num_classes = dataloaders.make_data_loader(args, **kwargs)
    args.num_classes = num_classes

elif args.dataset == 'nightcity':
    kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
    train_loader, val_loader,test_loader, num_classes = dataloaders.make_data_loader(args, **kwargs)
    args.num_classes = num_classes
else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))

if args.backbone == 'autodeeplab':
    model = Retrain_Autodeeplab(args)
else:
    raise ValueError('Unknown backbone: {}'.format(args.backbone))

# model = model.cuda()

from torchviz import make_dot
x = torch.zeros(4,3,512,512)
# model_arch = make_dot(model(x),params=dict(list(model.named_parameters())))
# model_arch.render('model_torchviz','./images',format='png')
from torchinfo import summary
# summary(model, (16, 3, 512, 512))
def visualize_image(self, writer, dataset, image, target, output, global_step):
    if (self.use_dist and dist.get_rank() == 0) or not self.use_dist:
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)


def save_image(dataset, image, target, output, step):
    grid_img = make_grid(image[:3].clone().cpu().data, 4, normalize=True)

    grid_out = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                   dataset=dataset), 4, normalize=False, range=(0, 255))

    grid_gt = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                   dataset=dataset), 4, normalize=False, range=(0, 255))

    plt.figure()
    plt.title('display')

    plt.subplot(311)
    plt.title('original_image')
    plt.imshow(grid_img)

    plt.subplot(312)
    plt.title('output_image')
    plt.imshow(grid_out)

    plt.subplot(312)
    plt.title('GT_image')
    plt.imshow(grid_gt)

    plt.show(block=True)
    plt.savefig('images/result/in_out_gt_'+step+'.png', bbox_inches='tight')









print(model.named_modules())
# summary = TensorboardSummary('./images')
# writer = summary.create_summary()
for ii, sample in enumerate(val_loader):
    for jj in range(sample["image"].size()[0]):
        img = sample['image'].numpy()
        gt = sample['label'].numpy()
        tmp = np.array(gt[jj]).astype(np.uint8)
        segmap = decode_segmap(tmp, dataset=args.dataset)
        img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
        img_tmp *= (0.229, 0.224, 0.225)
        img_tmp += (0.485, 0.456, 0.406)
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)
        plt.figure()
        plt.title('display')
        plt.subplot(211)
        plt.imshow(img_tmp)
        plt.subplot(212)
        plt.imshow(segmap)
    #
    if ii == 1:
        break

plt.show(block=True)
    # # Write image data to TensorBoard log dir
    # writer.add_image('input Images', inputs)
    # writer.add_image('mask Images', target)

