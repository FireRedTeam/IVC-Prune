import sys
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import warnings
from ..base import BaseModel
from ...smp import *
from PIL import Image
from ...dataset import DATASET_TYPE, DATASET_MODALITY
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


def bbox_to_tokens_vas(bbox, orig_width, orig_height, target_aspect_ratio, image_size=448, use_thumbnail=True):
    """
    将原图上的bbox转换为对应的token索引 (基于pad处理)
    
    Args:
        bbox: (x1, y1, x2, y2) 原图上的bbox坐标
        orig_width, orig_height: 原图尺寸
        target_aspect_ratio: (rows, cols) 切分的网格比例
        image_size: 每个block的尺寸
        use_thumbnail: 是否使用thumbnail
    
    Returns:
        token_info: 包含每个相关block的token信息
    """
    x1, y1, x2, y2 = bbox
    all_token_positions = []
    
    # token布局参数 (假设patch_size=32, downsample_ratio=2)
    h = w = 14  # math.ceil((image_size // patch_size) / downsample_ratio)
    
    # Global view: h * (w + 1) tokens
    global_tokens = h * (w + 1)
    
    # 分隔符: 1 token
    separator_tokens = 1
    
    # Local views dimensions
    num_height_tiles = target_aspect_ratio[1]  # 行数
    num_width_tiles = target_aspect_ratio[0]   # 列数
    
    # 1. 处理局部视图blocks中的bbox tokens
    # 计算pad后的目标尺寸
    target_width = image_size * target_aspect_ratio[0]  # 例如：448 * 2 = 896
    target_height = image_size * target_aspect_ratio[1]  # 例如：448 * 1 = 448
    
    # 计算pad的缩放比例和偏移
    scale = min(target_width / orig_width, target_height / orig_height)
    
    # 缩放后的实际图像尺寸
    scaled_width = orig_width * scale
    scaled_height = orig_height * scale
    
    # 计算居中pad的偏移量
    pad_left = (target_width - scaled_width) / 2
    pad_top = (target_height - scaled_height) / 2
    
    # 将bbox坐标从原图坐标系转换到pad后的坐标系
    padded_x1 = x1 * scale + pad_left
    padded_y1 = y1 * scale + pad_top
    padded_x2 = x2 * scale + pad_left
    padded_y2 = y2 * scale + pad_top
    
    # 确定bbox涉及的block范围
    block_x1 = int(padded_x1 // image_size)
    block_y1 = int(padded_y1 // image_size)
    block_x2 = int(padded_x2 // image_size)
    block_y2 = int(padded_y2 // image_size)
    
    # 确保不超出边界
    block_x1 = max(0, min(block_x1, target_aspect_ratio[0] - 1))
    block_y1 = max(0, min(block_y1, target_aspect_ratio[1] - 1))
    block_x2 = max(0, min(block_x2, target_aspect_ratio[0] - 1))
    block_y2 = max(0, min(block_y2, target_aspect_ratio[1] - 1))

    # 遍历涉及的每个block
    for block_y in range(block_y1, block_y2 + 1):
        for block_x in range(block_x1, block_x2 + 1):
            # 计算block在pad后图像中的边界
            block_left = block_x * image_size
            block_top = block_y * image_size
            
            # bbox在当前block内的坐标
            bbox_in_block_x1 = max(0, padded_x1 - block_left)
            bbox_in_block_y1 = max(0, padded_y1 - block_top)
            bbox_in_block_x2 = min(image_size, padded_x2 - block_left)
            bbox_in_block_y2 = min(image_size, padded_y2 - block_top)
            
            # 转换为14x14 token坐标
            token_scale = 14.0 / image_size
            
            token_x1 = int(bbox_in_block_x1 * token_scale)
            token_y1 = int(bbox_in_block_y1 * token_scale)
            token_x2 = int(bbox_in_block_x2 * token_scale)
            token_y2 = int(bbox_in_block_y2 * token_scale)
            
            # 确保在14x14范围内
            token_x1 = max(0, min(token_x1, 13))
            token_y1 = max(0, min(token_y1, 13))
            token_x2 = max(0, min(token_x2, 13))
            token_y2 = max(0, min(token_y2, 13))
            
            # 获取涉及的token索引
            for ty in range(token_y1, token_y2 + 1):
                for tx in range(token_x1, token_x2 + 1):
                    # 计算在local views中的索引
                    # Local views排列：(num_height_tiles * h) 行 x (num_width_tiles * w + 1) 列
                    # 每行包含所有tile在该行的tokens + 1个行分隔符
                    local_row = block_y * h + ty
                    local_col = block_x * w + tx
                    
                    # 在local views中的索引
                    local_index = local_row * (num_width_tiles * w + 1) + local_col
                    
                    # 在全局sequence中的索引
                    global_token_idx = global_tokens + separator_tokens + local_index
                    all_token_positions.append(global_token_idx)

    # 2. 如果使用thumbnail，计算bbox在thumbnail中的token位置
    if use_thumbnail:
        # thumbnail是整个原图pad到image_size x image_size
        thumb_scale = min(image_size / orig_width, image_size / orig_height)
        
        # 缩放后的尺寸
        thumb_scaled_width = orig_width * thumb_scale
        thumb_scaled_height = orig_height * thumb_scale
        
        # 计算居中pad的偏移量
        thumb_pad_left = (image_size - thumb_scaled_width) / 2
        thumb_pad_top = (image_size - thumb_scaled_height) / 2
        
        # bbox在thumbnail中的坐标
        thumb_x1 = x1 * thumb_scale + thumb_pad_left
        thumb_y1 = y1 * thumb_scale + thumb_pad_top
        thumb_x2 = x2 * thumb_scale + thumb_pad_left
        thumb_y2 = y2 * thumb_scale + thumb_pad_top
        
        # 转换为14x14 token坐标
        token_scale = 14.0 / image_size
        
        thumb_token_x1 = int(thumb_x1 * token_scale)
        thumb_token_y1 = int(thumb_y1 * token_scale)
        thumb_token_x2 = int(thumb_x2 * token_scale)
        thumb_token_y2 = int(thumb_y2 * token_scale)
        
        # 确保在14x14范围内
        thumb_token_x1 = max(0, min(thumb_token_x1, 13))
        thumb_token_y1 = max(0, min(thumb_token_y1, 13))
        thumb_token_x2 = max(0, min(thumb_token_x2, 13))
        thumb_token_y2 = max(0, min(thumb_token_y2, 13))
        
        # 获取thumbnail中涉及的token索引 (global view部分)
        for ty in range(thumb_token_y1, thumb_token_y2 + 1):
            for tx in range(thumb_token_x1, thumb_token_x2 + 1):
                # Global view: h行 x (w+1)列，每行有w个token + 1个行分隔符
                token_idx = ty * (w + 1) + tx
                all_token_positions.append(token_idx)

        all_token_positions.append(210)

        # for i in range(14):
        #     token_idx = i * 15 + 14 
        #     all_token_positions.append(token_idx)
        # all_token_positions.append(210)



    local_heights = num_height_tiles * 14
    local_widths = num_width_tiles *  14 + 1
    for i in range(local_heights):
        token_idx = (i+1) * local_widths - 1 + 211
        all_token_positions.append(token_idx)

    return all_token_positions



class DeepSeekVL2IVCP(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def check_install(self):
        try:
            import deepseek_vl2
        except Exception as e:
            logging.critical(
                'Please first install deepseek_vl2 from source codes in: https://github.com/deepseek-ai/DeepSeek-VL2')
            raise e

    def __init__(self, model_path='deepseek-ai/deepseek-vl2-tiny',fastv_config=None ,**kwargs):
        self.check_install()
        assert model_path is not None
        self.model_path = model_path
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

        self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.fastv_config = fastv_config


        model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path,
                                                                              config=config,
                                                                              trust_remote_code=True,
                                                                              torch_dtype=torch.bfloat16)
        self.model = model.cuda().eval()

        torch.cuda.empty_cache()
        default_kwargs = dict(max_new_tokens=2048, do_sample=False, use_cache=True)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def prepare_inputs(self, message, dataset=None):
        bbox = None
        if DATASET_TYPE(dataset) == 'VG':
            if dataset in {"RefCOCO_testA_foreground_deepseek", "RefCOCO_val_foreground_deepseek" ,"RefCOCO_testB_foreground_deepseek", "RefCOCO+_testA_foreground_deepseek", "RefCOCO+_testB_foreground_deepseek", "RefCOCO+_val_foreground_deepseek", "RefCOCOg_test_foreground_deepseek", "RefCOCOg_val_foreground_deepseek", "debug_foreground_deepseek"}:
                def prepare_itlist(msgs):
                    content, images = '', []
                    for s in msgs:
                        if s['type'] == 'image':
                            images.append(s['value'])
                            content += '<image>\nLocate <|ref|>'
                        elif s['type'] == 'text':

                            question = s['value'].split('@#@')[0]
                            content += question 
                            content += '<|/ref|> in the given image.'

                            bbox = np.array(s['value'].split('@#@')[1].split(' '), dtype=float)


                            content += s['value']
                    return content, images, bbox

                conversation = []
                if 'role' not in message[0]:
                    content, images, bbox = prepare_itlist(message)
                    conversation.append(dict(role='<|User|>', content=content, images=images))
                else:
                    role_map = {'user': '<|User|>', 'assistant': '<|Assistant|>'}
                    for msgs in message:
                        role = role_map[msgs['role']]
                        content, images, bbox = prepare_itlist(msgs['content'])
                        conversation.append(dict(role=role, content=content, images=images))
                conversation.append(dict(role='<|Assistant|>', content=''))

            else:
                # message [{'type': 'image', 'value': '/mnt/public/usr/sunzhichao/benchmark/images/debug/3.jpg'}, {'type': 'text', 'value': 'right man'}]
                def prepare_itlist(msgs):
                    content, images = '', []
                    for s in msgs:
                        if s['type'] == 'image':
                            images.append(s['value'])
                            content += '<image>\nLocate <|ref|>'
                        elif s['type'] == 'text':
                            content += s['value']
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

        return conversation, bbox

    def generate_inner(self, message, dataset=None):
        conversation, bbox = self.prepare_inputs(message, dataset)
        from deepseek_vl2.utils.io import load_pil_images
        pil_images = load_pil_images(conversation)
        # print(conversation)
        # print("message", message)
        if dataset == 'MMMU_DEV_VAL':
            if len(pil_images):
                h, w = pil_images[0].size
                pil_images[0] = pil_images[0].resize((2 * h, 2 * w), Image.BILINEAR)



        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        )
        all_token_positions = None
        if dataset in {"RefCOCO_testA_foreground_deepseek", "RefCOCO_val_foreground_deepseek" ,"RefCOCO_testB_foreground_deepseek", "RefCOCO+_testA_foreground_deepseek", "RefCOCO+_testB_foreground_deepseek", "RefCOCO+_val_foreground_deepseek", "RefCOCOg_test_foreground_deepseek", "RefCOCOg_val_foreground_deepseek", "debug_foreground_deepseek"}:
            width = pil_images[0].width
            height = pil_images[0].height
            all_token_positions = bbox_to_tokens_vas(bbox, width, height, prepare_inputs['images_spatial_crop'].squeeze().tolist())
        # print("prepare_inputs", prepare_inputs)
        prepare_inputs = prepare_inputs.to(self.model.device)


        inputs_embeds, (image_start_indices, image_token_lengths, positions_info) = self.model.prepare_inputs_embeds(**prepare_inputs)

        past_key_values = None
        if self.model.language.fastv_config != None:
            self.model.language.fastv_config['image_start_indices'] = image_start_indices
            self.model.language.fastv_config['image_token_lengths'] = image_token_lengths
        # inputs_embeds, past_key_values = self.model.incremental_prefilling(
        #     input_ids=prepare_inputs.input_ids,
        #     images=prepare_inputs.images,
        #     images_seq_mask=prepare_inputs.images_seq_mask,
        #     images_spatial_crop=prepare_inputs.images_spatial_crop,
        #     attention_mask=prepare_inputs.attention_mask,
        #     chunk_size=512
        # )
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
            image_pose=(all_token_positions, prepare_inputs['images_spatial_crop'].squeeze(), prepare_inputs['padding_token_positions']),
            **self.kwargs
        )

        answer = self.tokenizer.decode(
            outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
            skip_special_tokens=True
        )
        answer = answer.rstrip('.')
        if DATASET_TYPE(dataset) == "VG":
            bbox = extract_and_normalize_coordinates(answer)
            return bbox

        return answer

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message, dataset=dataset)

    def use_custom_prompt(self, dataset: str) -> bool:
        from vlmeval.dataset import DATASET_TYPE
        dataset_type = DATASET_TYPE(dataset, default=None)

        if dataset_type == 'VG':
            return True

        return False

    def build_prompt(self, line, dataset: str) -> list[dict[str,  str]]:
        from vlmeval.dataset import DATASET_TYPE
        dataset_type = DATASET_TYPE(dataset)
        if dataset_type == 'VG':
            return self._build_vg_prompt(line, dataset)

        raise ValueError(f'Unsupported dataset: {dataset}')


    def _build_vg_prompt(self, line, dataset: str) -> list[dict[str, str]]:

        tgt_path = toliststr(line['image_path'])

        question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]

        if dataset in {"RefCOCO_testA_foreground_deepseek", "RefCOCO_val_foreground_deepseek" ,"RefCOCO_testB_foreground_deepseek", "RefCOCO+_testA_foreground_deepseek", "RefCOCO+_testB_foreground_deepseek", "RefCOCO+_val_foreground_deepseek", "RefCOCOg_test_foreground_deepseek", "RefCOCOg_val_foreground_deepseek", "debug_foreground_deepseek"}:
            answer = line['answer']
            gt = np.array(answer.split(' '), dtype=float)
            msgs.append(dict(type='text', value=question+ '@#@' + answer))

            # msgs['text'].append(gt)
            # msgs.append(dict(type='bbox', value=answer))
            return msgs

        else:
            msgs.append(dict(type='text', value=question))

            return msgs