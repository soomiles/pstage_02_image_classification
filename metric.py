import warnings
import numpy as np
from sklearn.metrics import f1_score, classification_report
from scipy.stats.mstats import gmean
import torch
warnings.filterwarnings("ignore")


""" Loss Meter """
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


""" f1 Meter """
class F1Meter(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.y_trues = []
        self.y_preds = []
        self.scores = [0] * self.num_classes

    def update(self, y_pred, y_true):
        y_pred = y_pred.argmax(dim=1).data.cpu().numpy().astype(int)
        y_true = y_true.cpu().numpy().astype(int)
        self.y_trues.extend(y_true)
        self.y_preds.extend(y_pred)

    def calc_f1_score(self):
        cls_report = classification_report(
            y_true=self.y_trues,
            y_pred=self.y_preds,
            labels=np.arange(self.num_classes),
            output_dict=True,
        )
        for class_id in range(self.num_classes):
            self.scores[class_id] = cls_report[f'{class_id}']['f1-score']

    @property
    def avg(self):
        return gmean(self.scores)
