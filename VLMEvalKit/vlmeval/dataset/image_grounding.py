from .image_base import ImageBaseDataset
from ..smp import *
import numpy as np


def bbox_overlaps(bboxes1,
                  bboxes2,
                  mode='iou',
                  eps=1e-6,
                  use_legacy_coordinate=False):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (ndarray): Shape (n, 4)
        bboxes2 (ndarray): Shape (k, 4)
        mode (str): IOU (intersection over union) or IOF (intersection
            over foreground)
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.

    Returns:
        ious (ndarray): Shape (n, k)
    """

    assert mode in ['iou', 'iof']
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
        bboxes1[:, 3] - bboxes1[:, 1] + extra_length)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
        bboxes2[:, 3] - bboxes2[:, 1] + extra_length)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
            y_end - y_start + extra_length, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


class Grounding_COCO(ImageBaseDataset):
    TYPE = 'VG'

    DATASET_URL = {
        'RefCOCO_testA': '/mnt/public/usr/sunzhichao/benchmark/finetune_refcoco_testA.tsv',
        'RefCOCO_testB': '/PATH/refcoco_testB.tsv', 
        'RefCOCO_val': '/PATH/refcoco_val.tsv', 

        'RefCOCO+_testA': '/PATH/refcoco+_testA.tsv',
        'RefCOCO+_testB': '/PATH/refcoco+_testB.tsv',
        'RefCOCO+_val': '/PATH/refcoco+_val.tsv',

        'RefCOCOg_test': '/PATH/refcocog_test.tsv',
        'RefCOCOg_val': '/PATH/refcocog_val.tsv',

    }


    def load_data(self, dataset):
        data = super().load_data(dataset)
        return data


    @classmethod
    def evaluate(self, eval_file, **kwargs):
        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]

        ref, gt = {}, {}
        k = 1 
        TP = 0

        num = 0

        for i, line in enumerate(lines):
            
            num += 1 
            answer = line['answer']
            prediction = line['prediction']
            height = line['height']
            width = line['width']

            
            gt = np.array(answer.split(' '), dtype=float)
            pred = np.array([float(x.strip()) for x in prediction.strip('[]').split(',')])


            if gt.ndim == 1:
                gt = gt.reshape(1, -1)

            if pred.ndim == 1:
                pred = pred.reshape(1,  -1)

            if np.all(pred <= 1.1):  # 确保所有坐标都小于1
                # 将x坐标乘以宽度，y坐标乘以高度
                pred = pred.astype(float)  # 确保数据类型正确
                pred[:, [0, 2]] *= width   # 处理x_min和x_max
                pred[:, [1, 3]] *= height  # 处理y_min和y_max


            iou = bbox_overlaps(pred, gt)

            if max(iou[:k]) >= 0.5:
                TP += 1

        accuracy = TP / num
        score_pth = eval_file.replace('.xlsx', '_score.json')

        with open(score_pth, 'w') as f:
            json.dump({
                'accuracy': accuracy,
                'TP': TP,
                'number': num
            }, f)

        return {
                'accuracy': accuracy,
                'TP': TP,
                'number': num
            }