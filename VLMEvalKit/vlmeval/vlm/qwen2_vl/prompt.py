from __future__ import annotations
import numpy as np

def get_covered_patches(x, y, w, h, patch_size, num_cols, num_rows):
    """
    计算边界框覆盖的patch索引
    
    Args:
        x, y, w, h: 边界框坐标（左上角坐标和宽高）
        patch_size: patch大小
        num_cols, num_rows: patch网格的列数和行数
    
    Returns:
        list: 覆盖的patch索引列表 [(row, col), ...]
        list: 对应的token索引列表 [token_idx, ...]
    """
    # 计算边界框的左上角和右下角对应的patch坐标
    left_patch = int(x // patch_size)
    top_patch = int(y // patch_size)
    right_patch = min(int((x + w - 1) // patch_size), num_cols - 1)
    bottom_patch = min(int((y + h - 1) // patch_size), num_rows - 1)
    
    # 确保坐标在有效范围内
    left_patch = max(0, left_patch)
    top_patch = max(0, top_patch)
    right_patch = min(right_patch, num_cols - 1)
    bottom_patch = min(bottom_patch, num_rows - 1)
    
    patch_indices = []
    token_indices = []
    
    # 遍历覆盖的patch区域
    for row in range(top_patch, bottom_patch + 1):
        for col in range(left_patch, right_patch + 1):
            patch_indices.append((row, col))
            # token索引 = row * num_cols + col
            token_idx = row * num_cols + col
            token_indices.append(token_idx)
    
    return patch_indices, token_indices

class Qwen2VLPromptMixin:
    """
    Mixin class for Qwen2VLChat to build custom prompt for different datasets.

    Requires the following methods to be implemented in the subclass:
        - dump_image(line, dataset: str) -> str | list[str]

    Implements the following methods:
        - use_custom_prompt(dataset: str) -> bool
        - build_prompt(line, dataset: str) -> list[dict[str, str]]
    """

    def __init__(self, *args, use_custom_prompt: bool = True, is_qwen25: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._use_custom_prompt = use_custom_prompt
        self.is_qwen25 = is_qwen25

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset: str) -> bool:
        from vlmeval.dataset import DATASET_TYPE
        dataset_type = DATASET_TYPE(dataset, default=None)

        if not self._use_custom_prompt:
            return False
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return True
        if dataset_type == 'MCQ':
            return True
        if dataset_type == 'Y/N' and dataset in {'HallusionBench', 'POPE'}:  # MME has it's own prompt
            return True
        if dataset_type == 'VQA' and dataset not in {'MMVet'}:  # MMVet VQA has it's own prompt
            return True
        if dataset_type == 'VG':
            return True

        return False

    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        from vlmeval.dataset import DATASET_TYPE
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return self._build_mmmu_prompt(line, dataset)
        dataset_type = DATASET_TYPE(dataset, default=None)
        # print("build_prompt", dataset_type, dataset)
        if dataset in {'GQA_choose_all', 'GQA_choose_ChooseAttr', 'GQA_choose_ChooseCat', 'GQA_choose_LogicalObj', 'GQA_choose_QueryAttr', 'GQA_choose_ChooseRel', 'GQA_choose_CompareAttr'}:
            return self._build_gqa_debug_prompt(line, dataset)


        if dataset_type == 'MCQ':
            return self._build_mcq_prompt(line, dataset)
        if dataset_type == 'Y/N':
            return self._build_yorn_prompt(line, dataset)
        if dataset_type == 'VQA':
            return self._build_vqa_prompt(line, dataset)

        if dataset_type == 'VG':
            return self._build_vg_prompt(line, dataset)

        raise ValueError(f'Unsupported dataset: {dataset}')

    def _build_mmmu_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MMMU dataset: keep all images at beginning."""

        import string

        import pandas as pd

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_mcq_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MCQ dataset: use chinese prompt if the question contains chinese characters."""
        MCQ_CN_PROMPT = '请直接回答选项字母。'
        MCQ_EN_PROMPT = 'Please select the correct answer from the options above.'

        import string

        import pandas as pd

        def cn_string(s):
            import re

            if re.search('[\u4e00-\u9fff]', s):
                return True
            return False

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += MCQ_CN_PROMPT if cn_string(prompt) else MCQ_EN_PROMPT
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_yorn_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for YORN dataset:"""
        YORN_PROMPT = ' Please answer yes or no.'

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += YORN_PROMPT
        return msgs

    def _build_vqa_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for VQA dataset:"""
        VQA_PROMPT = '\nPlease try to answer the question with short words or phrases if possible.'

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += VQA_PROMPT
        return msgs

    def _build_gqa_debug_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for VQA dataset:"""
        VQA_PROMPT = '\nPlease try to answer the question with short words or phrases if possible.'
        obj_x = line['central object x']
        obj_y = line['central object y']
        obj_w = line['central object w']
        obj_h = line['central object h']
        dataset_w = line['image_w']
        dataset_h = line['image_h']
        patch_size = 28
        new_h = round(dataset_h / patch_size) * patch_size
        new_w = round(dataset_w / patch_size) * patch_size
        num_rows = new_h // patch_size
        num_cols = new_w // patch_size

        resize_scale_x = new_w / dataset_w
        resize_scale_y = new_h / dataset_h

        scale_x_to_original = dataset_w / new_w
        scale_y_to_original = dataset_h / new_h

        final_x = obj_x * scale_x_to_original * resize_scale_x
        final_y = obj_y * scale_y_to_original * resize_scale_y
        final_w = obj_w * scale_x_to_original * resize_scale_x
        final_h = obj_h * scale_y_to_original * resize_scale_y

        final_x = max(0, min(final_x, new_w - final_w))
        final_y = max(0, min(final_y, new_h - final_h))
        final_w = min(final_w, new_w - final_x)
        final_h = min(final_h, new_h - final_y)

        patch_indices, token_indices = get_covered_patches(
            final_x, final_y, final_w, final_h, patch_size, num_cols, num_rows
        )
        background_indices = [item for item in range(num_cols*num_rows) if item not in token_indices]

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += VQA_PROMPT

        msgs.append(dict(type='token_indices', value=[token_indices, background_indices]))
        return msgs

    def _build_vg_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """Build prompt for visual grounding dataset:
        Ask to locate an object and return its bbox coordinates in JSON format."""
        if self.is_qwen25 and dataset in {"RefCOCO_testA_foreground", "RefCOCO_testB_foreground", "RefCOCO_val_foreground", "RefCOCO+_testA_foreground", "RefCOCO+_testB_foreground", "RefCOCO+_val_foreground", "RefCOCOg_test_foreground", "RefCOCOg_val_foreground", "debug_foreground"}:
            VG_PROMPT = '\nLocate {} in this image and output the bbox coordinates in JSON format.'

            dataset_w = line['width']
            dataset_h = line['height']
            answer = line['answer']
            gt = np.array(answer.split(' '), dtype=float)
            patch_size = 28
            new_h = round(dataset_h / patch_size) * patch_size
            new_w = round(dataset_w / patch_size) * patch_size
            num_rows = new_h // patch_size
            num_cols = new_w // patch_size
            resize_scale_x = new_w / dataset_w
            resize_scale_y = new_h / dataset_h

            scale_x_to_original = dataset_w / new_w
            scale_y_to_original = dataset_h / new_h

            obj_x = gt[0]
            obj_y = gt[1]
            obj_w = gt[2] - gt[0]
            obj_h = gt[3] - gt[1]

            final_x = obj_x * scale_x_to_original * resize_scale_x
            final_y = obj_y * scale_y_to_original * resize_scale_y
            final_w = obj_w * scale_x_to_original * resize_scale_x
            final_h = obj_h * scale_y_to_original * resize_scale_y

            final_x = max(0, min(final_x, new_w - final_w))
            final_y = max(0, min(final_y, new_h - final_h))
            final_w = min(final_w, new_w - final_x)
            final_h = min(final_h, new_h - final_y)

            patch_indices, token_indices = get_covered_patches(
                final_x, final_y, final_w, final_h, patch_size, num_cols, num_rows
            )
            background_indices = [item for item in range(num_cols*num_rows) if item not in token_indices]

            tgt_path = self.dump_image(line, dataset)
            object_to_locate = line['question']  # Assuming 'question' contains the object to locate
            msgs = []


            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image', value=p) for p in tgt_path])
            else:
                msgs = [dict(type='image', value=tgt_path)]

            msgs.append(dict(type='text', value=VG_PROMPT.format(object_to_locate)))
            msgs.append(dict(type='token_indices', value=[token_indices, background_indices]))

        elif self.is_qwen25:

            # VG_PROMPT = '\nLocate all the {} and output the bbox coordinates in JSON format.'
            VG_PROMPT = '\nLocate {} and output the bbox coordinates in JSON format.'

            tgt_path = self.dump_image(line, dataset)
            object_to_locate = line['question']  # Assuming 'question' contains the object to locate
            msgs = []


            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image', value=p) for p in tgt_path])
            else:
                msgs = [dict(type='image', value=tgt_path)]

            # msgs.append(dict(type='text', value=VG_PROMPT.format(object_to_locate[:-1].lower())))
            msgs.append(dict(type='text', value=VG_PROMPT.format(object_to_locate)))

            # Add the prompt to locate the object
        
        else:
            # VG_PROMPT = '<|im_start|><|vision_start|><|image_pad|><|vision_end|>\n<|object_ref_start|>{}<|object_ref_end|>'
            # VG_PROMPT = '<|vision_start|><|image_pad|><|vision_end|>\n<|object_ref_start|>Detect {}<|object_ref_end|>'
            VG_PROMPT = '\n<|object_ref_start|>Detect {}<|object_ref_end|>'

            # VG_PROMPT = '\n<|object_ref_start|>{}<|object_ref_end|>'

            tgt_path = self.dump_image(line, dataset)
            object_to_locate = line['question']  # Assuming 'question' contains the object to locate
            msgs = []

            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image', value=p) for p in tgt_path])
            else:
                msgs = [dict(type='image', value=tgt_path)]

            msgs.append(dict(type='text', value=VG_PROMPT.format(object_to_locate[:-1].lower())))


        return msgs