import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import copy as cp
from .base import BaseModel
from ..smp import isimg, listinstr
from ..dataset import DATASET_TYPE
import re
import json


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



class QwenVL(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='Qwen/Qwen-VL', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eod_id
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda',trust_remote_code=True).eval()
        default_kwargs = dict(
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            min_new_tokens=1,
            num_return_sequences=1,
            use_cache=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        torch.cuda.empty_cache()

    def adjust_kwargs(self, dataset):
        kwargs = cp.deepcopy(self.kwargs)
        if DATASET_TYPE(dataset) in ['MCQ', 'Y/N']:
            kwargs['max_new_tokens'] = 32
        elif DATASET_TYPE(dataset) == 'Caption' and 'COCO' in dataset:
            kwargs['max_new_tokens'] = 32
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['OCRVQA', 'ChartQA', 'DocVQA'], dataset):
                kwargs['max_new_tokens'] = 100
            elif listinstr(['TextVQA'], dataset):
                kwargs['max_new_tokens'] = 10
        elif DATASET_TYPE(dataset) == 'VG':
            kwargs['max_new_tokens'] = 28
            kwargs['min_new_tokens'] = 10
            kwargs['length_penalty'] = 1

        
        return kwargs

    def generate_inner(self, message, dataset=None):
        if dataset is not None:
            kwargs = self.adjust_kwargs(dataset)
        else:
            kwargs = self.kwargs

        print(kwargs)
        if DATASET_TYPE(dataset) == 'VG':
            prompt = ''
            for s in message:
                if s['type'] == 'image':
                    prompt += f'<img>{s["value"]}</img>'
                elif s['type'] == 'text':
                    prompt += f'<ref>{s["value"]}</ref>'

        else:
            prompt = ''
            for s in message:
                if s['type'] == 'image':
                    prompt += f'<img>{s["value"]}</img>'
                elif s['type'] == 'text':
                    prompt += s['value']
            if dataset is not None and DATASET_TYPE(dataset) == 'VQA':
                prompt += ' Answer:'


        # 1111 <img>/mnt/public/usr/sunzhichao/benchmark/images/debug/36.jpg</img>right player
        # 2222 [{'type': 'image', 'value': '/mnt/public/usr/sunzhichao/benchmark/images/debug/36.jpg'}, {'type': 'text', 'value': 'right player'}]
        
        encoded = self.tokenizer([prompt], return_tensors='pt', padding='longest')
        input_ids = encoded.input_ids.to('cuda')
        attention_mask = encoded.attention_mask.to('cuda')

        pred = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs)
        answer = self.tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        
        if DATASET_TYPE(dataset) == 'VG':
            bbox = extract_and_normalize_coordinates(answer)
            return bbox

        return answer

    def generate_batch(self, messages_batch, dataset=None):

        batch_messages = []
        for message in messages_batch:
            if dataset is not None:
                kwargs = self.adjust_kwargs(dataset)
            else:
                kwargs = self.kwargs
            
            if DATASET_TYPE(dataset) == 'VG':
                prompt = ''
                for s in message:
                    if s['type'] == 'image':
                        prompt += f'<img>{s["value"]}</img>'
                    elif s['type'] == 'text':
                        prompt += f'<ref>{s["value"]}</ref>'

            else:
                prompt = ''
                for s in message:
                    if s['type'] == 'image':
                        prompt += f'<img>{s["value"]}</img>'
                    elif s['type'] == 'text':
                        prompt += s['value']
                if dataset is not None and DATASET_TYPE(dataset) == 'VQA':
                    prompt += ' Answer:'

            batch_messages.append(prompt)
            # 1111 <img>/mnt/public/usr/sunzhichao/benchmark/images/debug/36.jpg</img>right player
            # 2222 [{'type': 'image', 'value': '/mnt/public/usr/sunzhichao/benchmark/images/debug/36.jpg'}, {'type': 'text', 'value': 'right player'}]
        # print("!!!!!!", batch_messages)
        encoded = self.tokenizer(batch_messages, return_tensors='pt', padding='longest')
        input_ids = encoded.input_ids.to('cuda')
        attention_mask = encoded.attention_mask.to('cuda')

        preds = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs)
        # answer = self.tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        answers = [
            self.tokenizer.decode(pred[input_ids.size(1):].cpu(), skip_special_tokens=True).strip() for pred in preds
        ]

        # print(answers)

        if DATASET_TYPE(dataset) == 'VG':
            bboxes = [extract_and_normalize_coordinates(answer) for answer in answers]
            # bbox = extract_and_normalize_coordinates(answer)
            return bboxes

        return answers

class QwenVLChat(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='Qwen/Qwen-VL-Chat', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda', trust_remote_code=True).eval()
        torch.cuda.empty_cache()
        self.kwargs = kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def build_history(self, message):

        def concat_tilist(tilist):
            image_cnt = 1
            prompt = ''
            for item in tilist:
                if item['type'] == 'text':
                    prompt += item['value']
                elif item['type'] == 'image':
                    prompt += f"Picture {image_cnt}: <img>{item['value']}</img>\n"
                    image_cnt += 1
            return prompt

        assert len(message) % 2 == 0
        hist = []
        for i in range(len(message) // 2):
            m1, m2 = message[2 * i], message[2 * i + 1]
            assert m1['role'] == 'user' and m2['role'] == 'assistant'
            hist.append((concat_tilist(m1['content']), concat_tilist(m2['content'])))
        return hist

    def generate_inner(self, message, dataset=None):
        vl_list = [{'image': s['value']} if s['type'] == 'image' else {'text': s['value']} for s in message]
        query = self.tokenizer.from_list_format(vl_list)
        response, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)
        return response

    def chat_inner(self, message, dataset=None):
        assert len(message) % 2 == 1 and message[-1]['role'] == 'user'
        history = self.build_history(message[:-1])
        vl_list = [
            {'image': s['value']} if s['type'] == 'image' else {'text': s['value']}
            for s in message[-1]['content']
        ]
        query = self.tokenizer.from_list_format(vl_list)
        response, _ = self.model.chat(self.tokenizer, query=query, history=history, **self.kwargs)
        return response
