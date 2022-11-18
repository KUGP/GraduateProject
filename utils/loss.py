import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())


        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_index=255, cuda=False, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = thresh
        self.n_min = n_min
        self.ignore_lb = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index)
        if cuda:
            self.criteria = self.criteria.cuda()

    def forward(self, logits, labels):
        N, C, H, W = logits.size()   #16x19x769x769
        # print('#############')
        # print(logits.size())
        n_pixs = N * H * W #9,461,776    16x769x769
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C) # 9461776, 19
        # print(labels.size()) #[16, 769, 769, 3]
        labels = labels.view(-1) # 28385328
        # print('#########LOGITS AND LABELS SIZE ##########')
        # print(logits.size())
        # print(labels.size())
        with torch.no_grad():
            scores = F.softmax(logits, dim=1)
            labels_cpu = labels
            invalid_mask = labels_cpu == self.ignore_lb
            labels_cpu[invalid_mask] = 0
            picks = scores[torch.arange(n_pixs), labels_cpu]
            picks[invalid_mask] = 1
            sorteds, _ = torch.sort(picks)
            thresh = self.thresh if sorteds[self.n_min] < self.thresh else sorteds[self.n_min]
            labels[picks > thresh] = self.ignore_lb
        labels = labels.clone()
        loss = self.criteria(logits, labels)
        return loss

class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=False, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))




def build_criterion(args):
    print("=> Trying bulid {:}loss".format(args.criterion))
    if args.criterion == 'Ohem':
        return OhemCELoss(thresh=args.thresh, n_min=args.n_min, cuda=True)
    elif args.criterion == 'crossentropy':
        return SegmentationLosses(cuda=True).build_loss('ce')
    elif args.criterion == 'mixcrossentropy':
        return MixSoftmaxCrossEntropyLoss()
    else:
        raise ValueError('unknown criterion : {:}'.format(args.criterion))


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
