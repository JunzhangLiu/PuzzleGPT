import torch

def cal_precision_recall_f1_for_class(labels, preds, cls_id):
    preds_cls_idx = preds == cls_id
    labels_cls_idx = labels == cls_id
    tp = torch.logical_and(labels_cls_idx, preds_cls_idx).sum()
    predicted_positives = preds_cls_idx.sum()
    total_positives = labels_cls_idx.sum()
    precision = tp/predicted_positives
    recall = tp/total_positives
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def make_info_dict(labels, preds, classes, labels_to_idx_dict):
    info_dict = {}
    for cls in classes:
        prec, recal, f1 = cal_precision_recall_f1_for_class(
                                labels, preds, labels_to_idx_dict[cls])
        info_dict['{} Precision'.format(cls)] = prec
        info_dict['{} Recall'.format(cls)] = recal
        info_dict['{} F1'.format(cls)] = f1
    return info_dict


