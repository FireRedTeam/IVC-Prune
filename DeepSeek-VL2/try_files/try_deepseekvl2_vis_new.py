import deepseek_vl2
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from transformers import AutoModelForCausalLM, AutoConfig
from deepseek_vl2.utils.io import load_pil_images
import torch
from deepseek_vl2.serve.app_modules.utils import parse_ref_bbox
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

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
        "fastv_k": 2 ,
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


    # conversation = [
    #     {
    #         "role": "<|User|>",
    #         "content": "<image>\n<image>\n<|grounding|>In the first image, an object within the red rectangle is marked. Locate the object of the same category in the second image.",
    #         "images": [
    #             "images/incontext_visual_grounding_1.jpeg",
    #             "images/icl_vg_2.jpeg"
    #         ],
    #     },
    #     {"role": "<|Assistant|>", "content": ""},
    # ]


    # 这张图被编码成656个token
    # 如果用incremental_prefilling的方案，就按照chunk_size（512）裁成512+144 分开输入到模型中
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\nLocate <|ref|>guy on left of screen red shirt<|/ref|> in the given image.",
            "images": ["/mnt/public/usr/sunzhichao/VLMEvalKit/7836.jpg"]
        },
        {"role": "<|Assistant|>", "content": ""}

    ]


    pil_images = load_pil_images(conversation)

    prepare_inputs = vl_chat_processor.__call__(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device, dtype=dtype)

    # with torch.cuda.amp.autocast(enabled=False):

    with torch.no_grad():
        # inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
        #     input_ids=prepare_inputs.input_ids,
        #     images=prepare_inputs.images,
        #     images_seq_mask=prepare_inputs.images_seq_mask,
        #     images_spatial_crop=prepare_inputs.images_spatial_crop,
        #     attention_mask=prepare_inputs.attention_mask,
        #     chunk_size=51c2
        # )
        inputs_embeds, (image_start_indices, image_token_lengths, positions_info) = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        past_key_values = None
        if vl_gpt.language.fastv_config != None:
            vl_gpt.language.fastv_config['image_start_indices'] = image_start_indices
            vl_gpt.language.fastv_config['image_token_lengths'] = image_token_lengths
            # vl_gpt.language.fastv_config['positions_info'] = positions_info

        print("positions_info", positions_info)
        # 这个地方就已经生成了固定长度的 kv cache
        # for i, (key, value) in enumerate(past_key_values):
        #     print(f"Layer {i} - Key rope: {key.shape}, compressed kv : {value.shape}")        
        all_token_positions = None

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
            image_pose=(all_token_positions, prepare_inputs['images_spatial_crop'].squeeze(), prepare_inputs['padding_token_positions']),

            # output_hidden_states=True,
            # return_dict_in_generate=True
        )
        # all_hidden_states = outputs.hidden_states

        # for step_idx, step_hidden_states in enumerate(all_hidden_states):
        #     print(f"\n生成步骤 {step_idx + 1}:")
            
        #     # 遍历该步骤的所有解码层输出
        #     for layer_idx, layer_hidden in enumerate(step_hidden_states):
        #         print(f"  第 {layer_idx} 层隐藏状态维度: {layer_hidden.shape}")


        answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)
        print(f"{prepare_inputs['sft_format'][0]}", answer)
        vg_image = parse_ref_bbox(answer, image=pil_images[-1])

        print(vg_image)
        if vg_image is not None:
            vg_image.save("./vg.jpg", format="JPEG", quality=85)


        # [0, 231, 165, 748]