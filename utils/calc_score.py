# Adapted from score written by wkentaro, meetshah1995
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py


import numpy as np


class Score(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bitcount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            min_length=n_class **2
        ).reshape(n_class, n_class)

        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        accuracy = np.diag(hist).sum() / hist.sum()
        class_accuracy = np.diag(hist) / hist.sum(axis=1)
        mean_accuracy = np.nanmean(class_accuracy)
        class_iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.mean(class_iou)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * class_iou[freq > 0]).sum()
        class_iou = dict(zip(range(self.n_classes), class_iou))

        return(
            {
                "Overall Accuracy: \t": accuracy,
                "Mean Accuracy : \t": mean_accuracy,
                "FreqW Accuracy : \t": fwavacc,
                "Mean IoU : \t": mean_iou,
            }
            class_iou,
        )
