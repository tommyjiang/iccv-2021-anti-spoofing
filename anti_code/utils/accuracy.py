import torch

def get_accuracy(output, target, ignore_value= 100, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    batch_size = target.size(0)
    maxk = max(topk)
    with torch.no_grad():
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res