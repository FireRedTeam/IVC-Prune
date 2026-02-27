import deepseek_vl2
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from transformers import AutoModelForCausalLM, AutoConfig
from deepseek_vl2.utils.io import load_pil_images
import torch
from deepseek_vl2.serve.app_modules.utils import parse_ref_bbox
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'



#     return all_token_positions
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

        for i in range(14):
            token_idx = i * 15 + 14 
            all_token_positions.append(token_idx)
        all_token_positions.append(210)



    local_heights = num_height_tiles * 14
    local_widths = num_width_tiles *  14 + 1
    for i in range(local_heights):
        token_idx = (i+1) * local_widths - 1 + 211
        all_token_positions.append(token_idx)

    return all_token_positions



if __name__ == "__main__":

    model_path = "/mnt/public/usr/sunzhichao/hf_hub/models--deepseek-ai--deepseek-vl2-small"

    # model_path = "/mnt/public/usr/sunzhichao/hf_hub/models--deepseek-ai--deepseek-vl2-tiny"

    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
    dtype = torch.bfloat16

    tokenizer = vl_chat_processor.tokenizer

    # vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path,
    #                                                                           trust_remote_code=True,
    #                                                                           torch_dtype=dtype)
    fastv_config = {
        "use_fastv": True,
        "fastv_k": 1 ,
        "fastv_r": 0.5,
        "mask": False, 
        }
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.fastv_config = fastv_config
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                                  config=config,
                                                  trust_remote_code=True,
                                                  torch_dtype=dtype,
                                                #   fastv_config=fastv_config,
                                                #   _attn_implementation="flash_attention_2"
                                                  )

    vl_gpt = vl_gpt.cuda().eval()
    torch.cuda.empty_cache()

    # width = 1200 
    # height = 800 
    # white_image = Image.new('RGB', (width, height), (255, 255, 255))
    # white_image.save("white_1200_800.png")
    # 这张图被编码成656个token
    # 如果用incremental_prefilling的方案，就按照chunk_size（512）裁成512+144 分开输入到模型中
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\nLocate <|ref|>person sitting right<|/ref|> in the given image.",
            "images": ["/mnt/public/usr/sunzhichao/benchmark/images/finetune_refcoco_testA/1087.jpg"]
            # "images": ["/mnt/public/usr/sunzhichao/InternVL/internvl_chat/white.png"]
            # "images": ['/mnt/public/usr/sunzhichao/DeepSeek-VL2/white_1200_800.png']

        },
        {   "role": "<|Assistant|>", 
            "content": ""}

    ]


    bbox = [237.84, 348.33, 342.71000000000004, 541.3299999999999]

    # bbox = [360, 32, 899, 453]
    # bbox = [0, 147, 62, 482]

    pil_images = load_pil_images(conversation)

    prepare_inputs = vl_chat_processor.__call__(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device, dtype=dtype)
    print("prepare_inputs", prepare_inputs['images'].shape)  # prepare_inputs torch.Size([1, 5, 3, 384, 384])   [2, 2]
    # with torch.cuda.amp.autocast(enabled=False):

    pil_images_size = pil_images[0].size
    width = pil_images[0].width
    height = pil_images[0].height

    with torch.no_grad():
        # inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
        #     input_ids=prepare_inputs.input_ids,
        #     images=prepare_inputs.images,
        #     images_seq_mask=prepare_inputs.images_seq_mask,
        #     images_spatial_crop=prepare_inputs.images_spatial_crop,
        #     attention_mask=prepare_inputs.attention_mask,
        #     chunk_size=512
        # )
        print("prepare_inputs", prepare_inputs['images_spatial_crop'], prepare_inputs['images_spatial_crop'].shape, prepare_inputs['padding_token_positions'])

        all_token_positions = bbox_to_tokens_vas(bbox, width, height, prepare_inputs['images_spatial_crop'].squeeze().tolist())
        print("all_token_positions",  all_token_positions)


        inputs_embeds, (image_start_indices, image_token_lengths, positions_info) = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        past_key_values = None
        if vl_gpt.language.fastv_config != None:
            vl_gpt.language.fastv_config['image_start_indices'] = image_start_indices
            vl_gpt.language.fastv_config['image_token_lengths'] = image_token_lengths
            vl_gpt.language.fastv_config['positions_info'] = positions_info

        # 这个地方就已经生成了固定长度的 kv cache
        # for i, (key, value) in enumerate(past_key_values):
        #     print(f"Layer {i} - Key rope: {key.shape}, compressed kv : {value.shape}")        

        outputs = vl_gpt.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=past_key_values,

            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,

            # do_sample=False,
            # repetition_penalty=1.1,

            do_sample=False,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,
            use_cache=True,
            image_pose=(all_token_positions, prepare_inputs['images_spatial_crop'].squeeze(), prepare_inputs['padding_token_positions'], width, height)
            # output_hidden_states=True,
            # return_dict_in_generate=True
        )
        # all_hidden_states = outputs.hidden_states

        # for step_idx, step_hidden_states in enumerate(all_hidden_states):
        #     print(f"\n生成步骤 {step_idx + 1}:")
            
        #     # 遍历该步骤的所有解码层输出
        #     for layer_idx, layer_hidden in enumerate(step_hidden_states):
        #         print(f"  第 {layer_idx} 层隐藏状态维度: {layer_hidden.shape}")
        print("images_spatial_crop", prepare_inputs['images_spatial_crop'].squeeze())

        answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)
        print(f"{prepare_inputs['sft_format'][0]}", answer)
        vg_image = parse_ref_bbox(answer, image=pil_images[-1])

        print(vg_image)
        if vg_image is not None:
            vg_image.save("./vg.jpg", format="JPEG", quality=85)


        # [0, 231, 165, 748]