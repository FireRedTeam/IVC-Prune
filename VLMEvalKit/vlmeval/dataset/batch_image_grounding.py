import pandas as pd
from abc import abstractmethod
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



def img_root_map(dataset):
    if 'MM_NIAH' in dataset:
        return 'MMNIAH'
    if 'CRPE' in dataset:
        return 'CRPE'
    if 'OCRVQA' in dataset:
        return 'OCRVQA'
    if 'COCO_VAL' == dataset:
        return 'COCO'
    if 'MMMU' in dataset:
        return 'MMMU'
    if "QSpatial" in dataset:
        return "QSpatial"

    mmbench_root_map = {
        'MMBench_DEV_EN': 'MMBench', 'MMBench_TEST_EN': 'MMBench',
        'MMBench_DEV_CN': 'MMBench', 'MMBench_TEST_CN': 'MMBench',
        'MMBench': 'MMBench', 'MMBench_CN': 'MMBench',
        'MMBench_DEV_EN_V11': 'MMBench_V11', 'MMBench_TEST_EN_V11': 'MMBench_V11',
        'MMBench_DEV_CN_V11': 'MMBench_V11', 'MMBench_TEST_CN_V11': 'MMBench_V11',
        'MMBench_V11': 'MMBench', 'MMBench_CN_V11': 'MMBench',
    }
    if dataset in mmbench_root_map:
        return mmbench_root_map[dataset]
    return dataset

class Batch_Grounding_COCO:

    TYPE = 'VG'
    MODALITY = 'IMAGE'

    DATASET_URL = {
        'batch_RefCOCO_testA': '/mnt/public/usr/sunzhichao/benchmark/finetune_refcoco_testA.tsv',
        'batch_RefCOCO_testB': '/mnt/public/usr/sunzhichao/benchmark/finetune_refcoco_testB.tsv',

        'batch_RefCOCO+_testA': '/mnt/public/usr/sunzhichao/benchmark/finetune_refcoco+_testA.tsv',
        'batch_RefCOCO+_testB': '/mnt/public/usr/sunzhichao/benchmark/finetune_refcoco+_testB.tsv',
        'batch_RefCOCO+_val': '/mnt/public/usr/sunzhichao/benchmark/finetune_refcoco+_val.tsv',

        'batch_RefCOCOg_test': '/mnt/public/usr/sunzhichao/benchmark/finetune_refcocog_test.tsv',
        'batch_RefCOCOg_val': '/mnt/public/usr/sunzhichao/benchmark/finetune_refcocog_val.tsv',
        'batch_debug': '/mnt/public/usr/sunzhichao/benchmark/finetune_refcoco_testA_try100.tsv'
    }
    DATASET_MD5 = {}
    
    def __init__(self, dataset='MMBench', skip_noimg=True, batch_size=4):
        """
        初始化数据集
        
        Args:
            dataset: 数据集名称
            skip_noimg: 是否跳过没有图片的样本
            batch_size: 默认的批处理大小
        """
        self.batch_size = batch_size
        ROOT = LMUDataRoot()
        self.dataset_name = dataset
        self.img_root = osp.join(ROOT, 'images', img_root_map(dataset))

        # 原有代码保持不变...
        data = self.load_data(dataset)
        self.skip_noimg = skip_noimg
        if skip_noimg and 'image' in data:
            data = data[~pd.isna(data['image'])]

        data['index'] = [str(x) for x in data['index']]

        self.meta_only = True

        # The image field can store the base64 encoded image or another question index (for saving space)
        if 'image' in data:
            data['image'] = [str(x) for x in data['image']]
            image_map = {x: y for x, y in zip(data['index'], data['image'])}
            for k in image_map:
                if len(image_map[k]) <= 64:
                    idx = image_map[k]
                    assert idx in image_map and len(image_map[idx]) > 64
                    image_map[k] = image_map[idx]

            images = [toliststr(image_map[k]) for k in data['index']]
            data['image'] = [x[0] if len(x) == 1 else x for x in images]
            self.meta_only = False

        if 'image_path' in data:
            paths = [toliststr(x) for x in data['image_path']]
            data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]

        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        self.data = data
        self.post_build(dataset)

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     print("!!!!!", idx)
    #     if isinstance(idx, (list, tuple, np.ndarray)):
    #         return self.get_batch(idx)
    #     return dict(self.data.iloc[idx])

    def get_batch(self, indices):
        """
        获取指定索引的批数据
        
        Args:
            indices: 索引列表
            
        Returns:
            包含每个索引对应数据的字典列表
        """
        return [dict(self.data.iloc[i]) for i in indices]
    
    def prepare_tsv(self, url, file_md5=None):
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        update_flag = False
        file_name = url.split('/')[-1]
        data_path = osp.join(data_root, file_name)
        self.data_path = data_path
        if osp.exists(data_path) and (file_md5 is None or md5(data_path) == file_md5):
            pass
        else:
            warnings.warn('The dataset tsv is not downloaded')
            download_file(url, data_path)
            update_flag = True

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None) or update_flag:
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)
    
    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                assert 'image_path' in line
                for img, im_name in zip(line['image'], line['image_path']):
                    path = osp.join(self.img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)
            else:
                tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])

        return tgt_path

    def display(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        assert isinstance(line, pd.Series) or isinstance(line, dict)
        mmqa_display(line)

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    # Post built hook, will be called after the dataset is built, can override
    def post_build(self, dataset):
        pass


    # Given the dataset name, return the dataset as a pandas dataframe, can override
    def load_data(self, dataset):
        url = self.DATASET_URL.get(dataset, None)
        if url is None or url == '':
            url = dataset + '.tsv'
        file_md5 = self.DATASET_MD5[dataset] if dataset in self.DATASET_MD5 else None
        return self.prepare_tsv(url, file_md5)


    def get_batches(self, indices=None, batch_size=None):
        """
        将数据集或指定的索引划分为批次
        
        Args:
            indices: 要处理的索引列表，如果为None则使用整个数据集
            batch_size: 批大小，如果为None则使用默认批大小
            
        Returns:
            批索引的列表
        """

        print("!!!!2222", indices, batch_size )
        if batch_size is None:
            batch_size = self.batch_size
            
        if indices is None:
            indices = list(range(len(self.data)))
            
        # 将索引划分为批次
        num_batches = (len(indices) + batch_size - 1) // batch_size
        batches = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            batches.append(batch_indices)
            
        return batches
    
    def estimate_optimal_batch_size(self, gpu_memory_gb=16, safety_factor=0.7):
        """
        估算基于GPU内存的最佳批大小
        
        Args:
            gpu_memory_gb: GPU内存大小（GB）
            safety_factor: 安全系数，避免使用全部GPU内存
            
        Returns:
            估计的最佳批大小
        """
        try:
            # 这里的逻辑是非常简化的，需要根据实际模型和数据调整
            # 假设每个图像样本大约需要2GB内存
            available_memory = gpu_memory_gb * safety_factor
            estimated_batch_size = max(1, int(available_memory / 2))
            return estimated_batch_size
        except:
            # 如果无法估算，返回默认值
            return self.batch_size

    def build_batch_prompt(self, lines_or_indices, batch_size=None):
        """
        为批数据构建提示
        
        Args:
            lines_or_indices: 数据记录列表或索引列表
            batch_size: 批大小，如果提供则将输入分成多个批次处理
            
        Returns:
            提示列表的列表，每个提示包含多模态消息
        """
        # 处理索引或数据记录
        if all(isinstance(x, int) for x in lines_or_indices):
            # 如果是索引列表
            if batch_size is not None and batch_size > 0:
                # 如果提供了batch_size，将索引分成批次处理
                batches = self.get_batches(lines_or_indices, batch_size)
                all_prompts = []
                for batch_indices in batches:
                    batch_data = self.get_batch(batch_indices)
                    batch_prompts = self._build_prompts_for_records(batch_data)
                    all_prompts.extend(batch_prompts)
                return all_prompts
            else:
                # 直接处理所有索引
                batch_data = self.get_batch(lines_or_indices)
                return self._build_prompts_for_records(batch_data)
        else:
            # 如果是数据记录列表
            return self._build_prompts_for_records(lines_or_indices)
    
    def _build_prompts_for_records(self, records):
        """
        为数据记录列表构建提示
        
        Args:
            records: 数据记录列表
            
        Returns:
            提示列表的列表
        """
        batch_prompts = []
        for record in records:
            if self.meta_only:
                tgt_path = toliststr(record['image_path'])
            else:
                tgt_path = self.dump_image(record)

            question = record['question']

            msgs = []
            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image', value=p) for p in tgt_path])
            else:
                msgs = [dict(type='image', value=tgt_path)]
            msgs.append(dict(type='text', value=question))
            batch_prompts.append(msgs)
        
        return batch_prompts
        
    # 为单个样本构建提示（保持向后兼容）
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        
        prompts = self._build_prompts_for_records([line])
        return prompts[0]


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

            if np.all(pred < 1.0):  # 确保所有坐标都小于1
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

        # coco_caption_score_dict = scorer.compute_scores()
        # dump(coco_caption_score_dict, score_pth)
        return {
                'accuracy': accuracy,
                'TP': TP,
                'number': num
            }