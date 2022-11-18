import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image

### For DEMO video_image loader transforms ::: No seg_mask only input image

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        # mask = sample['label']
        img = np.array(img).astype(np.float32)
        # mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()

        return {'image': img}

class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img}

# resize to 512*1024
class FixedResize(object):
    """change the short edge length to size"""

    def __init__(self, resize=512):
        self.size1 = resize  # size= 512

    def __call__(self, sample):
        img = sample['image']

        w, h = img.size
        if w > h:
            oh = self.size1
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.size1
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        return {'image': img}

class transform_val(object):
    def __init__(self, args, mean, std):
        self.composed_transforms = transforms.Compose([
            FixedResize(resize=args.crop_size),
            Normalize(mean, std),
            ToTensor()])

    def __call__(self, sample):
        return self.composed_transforms(sample)


