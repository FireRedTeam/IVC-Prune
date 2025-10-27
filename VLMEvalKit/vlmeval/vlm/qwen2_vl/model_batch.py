from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_rank_and_world_size, get_gpu_memory, auto_split_flag, listinstr
import json
import re
import time


def extract_bbox_coordinates(response_str):
    """
    使用正则表达式从字符串中提取bbox_2d坐标
    
    Args:
        response_str (str): 包含边界框信息的字符串
        
    Returns:
        list: 包含边界框坐标的列表 [x1, y1, x2, y2]
        如果没找到匹配项，返回[0, 0, 0, 0]
    """
    # 正则表达式匹配 "bbox_2d": [数字, 数字, 数字, 数字]
    pattern = r'"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    
    match = re.search(pattern, response_str)
    if match:
        # 提取四个坐标值并转换为整数
        x1 = int(match.group(1))
        y1 = int(match.group(2))
        x2 = int(match.group(3))
        y2 = int(match.group(4))
        return [x1, y1, x2, y2]
    else:
        print("未找到边界框坐标，返回[0, 0, 0, 0]")
        return [0, 0, 0, 0]


def extract_and_normalize_coordinates(box_string):
    """
    从Qwen模型返回的坐标字符串中提取坐标并归一化到[0,1]范围
    
    参数:
        box_string: 包含坐标的字符串，如'<|box_start|>(69,306),(250,988)<|box_end|><|im_end|>'
        
    返回:
        归一化后的坐标元组：((x1_norm, y1_norm), (x2_norm, y2_norm))
        如果未找到坐标，则返回None
    """
    # 使用正则表达式提取坐标
    pattern = r'\((\d+),(\d+)\),\((\d+),(\d+)\)'
    match = re.search(pattern, box_string)
    
    if match:
        # 提取坐标值
        x1, y1, x2, y2 = map(int, match.groups())
        
        # 归一化到[0,1]范围
        x1_norm = x1 / 1000.0
        y1_norm = y1 / 1000.0
        x2_norm = x2 / 1000.0
        y2_norm = y2 / 1000.0
        
        return [x1_norm, y1_norm, x2_norm, y2_norm]
    else:
        return  [0, 0, 0, 0]


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def split_model():
    device_map = {}

    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = total_gpus // world_size
    # + 8 is virtual layers for the memory of visual
    num_layers = 80 + 8
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2
    layer_cnt = 0

    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = rank + i * world_size
            layer_cnt += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map['visual'] = rank
    device_map['model.embed_tokens'] = rank
    device_map['model.norm'] = last_gpu
    device_map['model.rotary_emb'] = last_gpu
    device_map['lm_head'] = last_gpu
    return device_map


class Qwen2VLChatBatch(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        visual_grounding: bool = False
    ):

        self.supports_batch = True
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        rank, world_size = get_rank_and_world_size()
        assert model_path is not None
        self.model_path = model_path
        self.visual_grounding = visual_grounding
        MODEL_CLS = None

        self.is_qwen25 = False

        if listinstr(['2.5', '2_5', 'qwen25'], model_path.lower()):
            self.is_qwen25 = True
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            self.is_qwen25 = False
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        self.processor.tokenizer.padding_side = "left"

        super().__init__(use_custom_prompt=use_custom_prompt, is_qwen25=self.is_qwen25)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        # If only one process and GPU memory is less than 40GB
        if '72b' in self.model_path.lower():
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map=split_model(), attn_implementation='flash_attention_2'
            )
            self.model.eval()
        elif auto_split_flag():
            assert world_size == 1, 'Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL'
            # Will Use All GPUs to run one model
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
            )
        else:
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map='cpu', attn_implementation='flash_attention_2'
            )
            self.model.cuda().eval()

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content


    def generate_inner(self, messages_batch, dataset=None):
        """
        Process a batch of messages for inference
        
        Args:
            messages_batch: List of message lists or a single message list
            dataset: Optional dataset name for special processing
        
        Returns:
            List of responses or bounding boxes depending on the mode
        """
        # Handle the case of a single message (backward compatibility)
        is_single_message = not isinstance(messages_batch[0], list)
        if is_single_message:
            messages_batch = [messages_batch]
        
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err
        # Handle visual grounding for Qwen < 2.5 (non-batch mode)
        if self.visual_grounding and not self.is_qwen25 and len(messages_batch) == 1:
            # Process single item using existing code
            messages = []
            messages.append({'role': 'user', 'content': self._prepare_content(messages_batch[0], dataset=dataset)})

            images, videos = process_vision_info(messages)
            text = messages_batch[0][-1]['value']
            inputs = self.processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt')

            inputs = inputs.to('cuda')
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            out = self.processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            response = out[0]
            bbox = extract_and_normalize_coordinates(response)
            return bbox if is_single_message else [bbox]
        
        # Prepare messages for batch processing
        batch_messages = []
        for message in messages_batch:
            msg = []
            if self.system_prompt is not None:
                msg.append({'role': 'system', 'content': self.system_prompt})
            msg.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
            batch_messages.append(msg)
        
        if self.verbose:
            print(f'\033[31m{batch_messages}\033[0m')
        
        # Apply chat template to each conversation
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]
        
        # Process vision info for all messages
        images, videos = process_vision_info(batch_messages)
        
        # Create batch inputs
        inputs = self.processor(text=texts, images=images, videos=videos, padding=True, return_tensors='pt')
        inputs = inputs.to('cuda')
        
        # Generate responses
        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        
        # Extract the generated text (removing prompt)
        generated_ids_trimmed = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated text
        responses = self.processor.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Post-process the responses if needed
        if self.post_process:
            for i, response in enumerate(responses):
                resp = response.split('\\boxed{')[-1]
                lt = len(resp)
                counter, end = 1, None
                for j in range(lt):
                    if resp[j] == '{':
                        counter += 1
                    elif resp[j] == '}':
                        counter -= 1
                    if counter == 0:
                        end = j
                        break
                    elif j == lt - 1:
                        end = lt
                        break
                if end is not None:
                    responses[i] = resp[:end]
        
        if self.verbose:
            print(f'\033[32m{responses}\033[0m')
        
        # For visual grounding with Qwen 2.5
        if self.visual_grounding and self.is_qwen25:
            bboxes = [extract_bbox_coordinates(response) for response in responses]
            if self.verbose and any(bboxes):
                print(f"Extracted bounding boxes: {bboxes}")
            return bboxes[0] if is_single_message else bboxes
        
        # Return the response (single response or batch)
        return responses[0] if is_single_message else responses


    def generate_batch(self, messages_batch, dataset=None):
        """
        Process a batch of messages for inference
        
        Args:
            messages_batch: List of message lists or a single message list
            dataset: Optional dataset name for special processing
        
        Returns:
            List of responses or bounding boxes depending on the mode
        """
        # Handle the case of a single message (backward compatibility)
        is_single_message = not isinstance(messages_batch[0], list)
        if is_single_message:
            messages_batch = [messages_batch]
        
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err
        # Handle visual grounding for Qwen < 2.5 (non-batch mode)
        if self.visual_grounding and not self.is_qwen25 and len(messages_batch) == 1:
            # Process single item using existing code
            messages = []
            messages.append({'role': 'user', 'content': self._prepare_content(messages_batch[0], dataset=dataset)})

            images, videos = process_vision_info(messages)

            print(messages)
            text = messages_batch[0][-1]['value']
            inputs = self.processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt')

            inputs = inputs.to('cuda')
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            out = self.processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            response = out[0]
            bbox = extract_and_normalize_coordinates(response)

            return bbox if is_single_message else [bbox]
        
        if self.visual_grounding and not self.is_qwen25:
            batch_messages = []
            batch_texts = []
            for message in messages_batch:
                messages = []
                messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
                # print(messages)
                text = messages[0]['content'][1]['text']

                batch_messages.append(messages)
                batch_texts.append(text)

            images, videos = process_vision_info(batch_messages)

            inputs = self.processor(text=batch_texts, images=images, videos=videos, padding=True, return_tensors='pt')
            inputs = inputs.to('cuda')
        
            # Generate responses
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128
            )
            generated_ids_trimmed = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            responses = self.processor.tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )

            bboxes = [extract_and_normalize_coordinates(response) for response in responses]
            print(bboxes[-1])
            return bboxes

        # Prepare messages for batch processing
        batch_messages = []
        for message in messages_batch:
            msg = []
            if self.system_prompt is not None:
                msg.append({'role': 'system', 'content': self.system_prompt})
            msg.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
            batch_messages.append(msg)
        
        if self.verbose:
            print(f'\033[31m{batch_messages}\033[0m')
        
        # Apply chat template to each conversation
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]
        
        # Process vision info for all messages
        images, videos = process_vision_info(batch_messages)
        
        # Create batch inputs
        inputs = self.processor(text=texts, images=images, videos=videos, padding=True, return_tensors='pt')
        inputs = inputs.to('cuda')
       
        # Generate responses
        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        
        # Extract the generated text (removing prompt)
        generated_ids_trimmed = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated text
        responses = self.processor.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Post-process the responses if needed
        if self.post_process:
            for i, response in enumerate(responses):
                resp = response.split('\\boxed{')[-1]
                lt = len(resp)
                counter, end = 1, None
                for j in range(lt):
                    if resp[j] == '{':
                        counter += 1
                    elif resp[j] == '}':
                        counter -= 1
                    if counter == 0:
                        end = j
                        break
                    elif j == lt - 1:
                        end = lt
                        break
                if end is not None:
                    responses[i] = resp[:end]
        
        if self.verbose:
            print(f'\033[32m{responses}\033[0m')
        # For visual grounding with Qwen 2.5
        if self.visual_grounding and self.is_qwen25:
            bboxes = [extract_bbox_coordinates(response) for response in responses]
            if self.verbose and any(bboxes):
                print(f"Extracted bounding boxes: {bboxes}")
            return bboxes[0] if is_single_message else bboxes
        
        # elif self.visual_grounding and not self.is_qwen25:
        #     bboxes = [extract_and_normalize_coordinates(response) for response in responses]
        #     print(bboxes)
        #     return bboxes
        # Return the response (single response or batch)
        return responses[0] if is_single_message else responses
