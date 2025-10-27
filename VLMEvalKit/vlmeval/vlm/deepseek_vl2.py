import sys
import torch
from transformers import AutoModelForCausalLM
import warnings
from .base import BaseModel
from ..smp import *
from PIL import Image
from ..dataset import DATASET_TYPE, DATASET_MODALITY
import re

def extract_and_normalize_coordinates(box_string):
    """
    从InternVL模型返回的坐标字符串中提取坐标并归一化到[0,1]范围
    
    参数:
        box_string: 包含坐标的字符串，如'[123,456,789,012]' 或 '[[123,456,789,012]]'
        
    返回:
        归一化后的坐标列表：[x1_norm, y1_norm, x2_norm, y2_norm]
        如果未找到坐标，则返回[0, 0, 0, 0]
    """
    # 使用正则表达式提取坐标（参考原代码的PATTERN）
    pattern = r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*'
    match = re.search(pattern, box_string)
    
    if match:
        try:
            # 提取坐标值
            x1, y1, x2, y2 = map(float, match.groups())
            
            # 构建坐标元组用于判断是否需要归一化
            coords_sum = x1 + y1 + x2 + y2
            
            # 如果坐标值较大（sum >= 4），则除以1000进行归一化
            if coords_sum >= 4:
                x1_norm = x1 / 1000.0
                y1_norm = y1 / 1000.0
                x2_norm = x2 / 1000.0
                y2_norm = y2 / 1000.0
            else:
                x1_norm = x1
                y1_norm = y1
                x2_norm = x2
                y2_norm = y2
            
            return [x1_norm, y1_norm, x2_norm, y2_norm]
        except:
            return [0, 0, 0, 0]
    else:
        return [0, 0, 0, 0]

class DeepSeekVL2(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def check_install(self):
        try:
            import deepseek_vl2
        except Exception as e:
            logging.critical(
                'Please first install deepseek_vl2 from source codes in: https://github.com/deepseek-ai/DeepSeek-VL2')
            raise e

    def __init__(self, model_path='deepseek-ai/deepseek-vl2-tiny', **kwargs):
        self.check_install()
        assert model_path is not None
        self.model_path = model_path
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

        self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path,
                                                                              trust_remote_code=True,
                                                                              torch_dtype=torch.bfloat16)
        self.model = model.cuda().eval()

        torch.cuda.empty_cache()
        default_kwargs = dict(max_new_tokens=2048, do_sample=False, use_cache=True)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def prepare_inputs(self, message, dataset=None):

        if DATASET_TYPE(dataset) == 'VG':
            # message [{'type': 'image', 'value': '/mnt/public/usr/sunzhichao/benchmark/images/debug/3.jpg'}, {'type': 'text', 'value': 'right man'}]
            def prepare_itlist(msgs):
                content, images = '', []
                for s in msgs:
                    if s['type'] == 'image':
                        images.append(s['value'])
                        content += '<image>\nLocate <|ref|> '
                    elif s['type'] == 'text':
                        content += s['value'][:-1].lower()
                        content += '<|/ref|> in the given image.'
                return content, images

            conversation = []
            if 'role' not in message[0]:
                content, images = prepare_itlist(message)
                conversation.append(dict(role='<|User|>', content=content, images=images))
            else:
                role_map = {'user': '<|User|>', 'assistant': '<|Assistant|>'}
                for msgs in message:
                    role = role_map[msgs['role']]
                    content, images = prepare_itlist(msgs['content'])
                    conversation.append(dict(role=role, content=content, images=images))
            conversation.append(dict(role='<|Assistant|>', content=''))
            # conversation [{'role': '<|User|>', 'content': '<image>\n<|ref|>man on right<|ref|>', 'images': ['/mnt/public/usr/sunzhichao/benchmark/images/debug/2.jpg']}, {'role': '<|Assistant|>', 'content': ''}]            exit()

        elif dataset == 'MMMU_DEV_VAL':

            def prepare_itlist(msgs):
                content, images = '', []
                image_idx = 1
                for s in msgs:
                    if s['type'] == 'image':
                        images.append(s['value'])
                        content += f'<image {image_idx}>'
                        image_idx += 1
                    elif s['type'] == 'text':
                        content += s['value']
                # content = '<image>' * (image_idx-1) + '\n' + content
                content = '<image>' * (image_idx - 1) + '\n' + content
                return content, images

            conversation = []
            if 'role' not in message[0]:
                content, images = prepare_itlist(message)
                content = content.replace(
                    'Please select the correct answer from the options above.',
                    "Answer with the option's letter from the given choices directly. Answer the question using a single word or phrase.\n"  # noqa
                )
                content = content.replace('Question:', "")
                content = content.replace('Options:\n', "")
                conversation.append(dict(role='<|User|>', content=content, images=images))
            else:
                role_map = {'user': '<|User|>', 'assistant': '<|Assistant|>'}
                for msgs in message:
                    role = role_map[msgs['role']]
                    content, images = prepare_itlist(msgs['content'])
                    content = content.replace(
                        'Please select the correct answer from the options above.',
                        "Answer with the option's letter from the given choices directly. Answer the question using a single word or phrase.\n"  # noqa
                    )
                    content = content.replace('Question:', "")
                    content = content.replace('Options:\n', "")
                    conversation.append(dict(role=role, content=content, images=images))
            conversation.append(dict(role='<|Assistant|>', content=''))

        else:

            def prepare_itlist(msgs):
                content, images = '', []
                for s in msgs:
                    if s['type'] == 'image':
                        images.append(s['value'])
                        content += '<image>\n'
                    elif s['type'] == 'text':
                        content += s['value']
                return content, images

            conversation = []
            if 'role' not in message[0]:
                content, images = prepare_itlist(message)
                conversation.append(dict(role='<|User|>', content=content, images=images))
            else:
                role_map = {'user': '<|User|>', 'assistant': '<|Assistant|>'}
                for msgs in message:
                    role = role_map[msgs['role']]
                    content, images = prepare_itlist(msgs['content'])
                    conversation.append(dict(role=role, content=content, images=images))
            conversation.append(dict(role='<|Assistant|>', content=''))

        return conversation

    def generate_inner(self, message, dataset=None):
        conversation = self.prepare_inputs(message, dataset)
        from deepseek_vl2.utils.io import load_pil_images
        pil_images = load_pil_images(conversation)

        if dataset == 'MMMU_DEV_VAL':
            if len(pil_images):
                h, w = pil_images[0].size
                pil_images[0] = pil_images[0].resize((2 * h, 2 * w), Image.BILINEAR)
        # print("conversation", conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        )
        # print("prepare_inputs", prepare_inputs)
        prepare_inputs = prepare_inputs.to(self.model.device)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        inputs_embeds, past_key_values = self.model.incremental_prefilling(
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            chunk_size=512
        )

        # run the model to get the response
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=past_key_values,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.kwargs
        )

        answer = self.tokenizer.decode(
            outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
            skip_special_tokens=True
        )
        answer = answer.rstrip('.')

        # return answer
        # # print(answer)
        if DATASET_TYPE(dataset) == "VG":
            bbox = extract_and_normalize_coordinates(answer)
            print(bbox)
            return bbox

        return answer

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message, dataset=dataset)
