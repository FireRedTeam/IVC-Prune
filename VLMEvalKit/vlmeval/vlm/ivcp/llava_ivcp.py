import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
from ...dataset import DATASET_TYPE, DATASET_MODALITY
import copy
import requests


class LLaVAIVCP(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, 
                 model_path="llava-hf/llava_v1.5_7b-hf",
                 fastv_config = None,
                 drop_config = None,
                 **kwargs):

        from transformers import AutoProcessor, LlavaForConditionalGeneration
        self.system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.model_path = model_path

        # fastv_config = {
        #     "use_fastv": True,
        #     "fastv_k": 2,
        #     "fastv_r": 0.5,
        #     "image_token_start_index": 5, 
        #     "image_token_length": 576 
        # } 

        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            # attn_implementation="eager",

            fastv_config = fastv_config, 
            )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            # revision="a272c74"
            )
        # processor = AutoProcessor.from_pretrained(model_id)

        model = model.eval()
        self.model = model.cuda()

        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            # min_new_tokens=200, 
            max_new_tokens=100,
            top_p=None,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True
        )  # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )

        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message

    def concat_tilist(self, message):
        text, images = "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                text += " <image> "
                images.append(item["value"])
        return text, images

    def generate_inner(self, message, dataset=None):

        # print("message", message)
        # exit()
        if dataset in {'GQA_choose_ChooseAttr', 'GQA_choose_ChooseCat', 'GQA_choose_LogicalObj', 'GQA_choose_QueryAttr', 'GQA_choose_ChooseRel', 'GQA_choose_CompareAttr'}:
            token_indices = None
            for item in message:
                if item['type'] == 'token_indices':
                    token_indices = item['value']
                    break
            content, images = self.concat_tilist(message)

            images = [Image.open(s).convert("RGB") for s in images]
            prompt = self.system_prompt + "USER: " + content + " ASSISTANT: "
            inputs = self.processor(prompt, images, return_tensors='pt').to(
                "cuda", dtype=torch.float16
            )
            output_ids = self.model.generate(
                    **inputs,
                    token_indices=token_indices,
                    **self.kwargs
            )

            full_response = self.processor.batch_decode(output_ids.sequences,skip_special_tokens=True)[
                0
            ].strip()
            answer = full_response.split("ASSISTANT:")[1].strip()  # 只保留ASSISTANT部分

            return answer

        else:
            content, images = self.concat_tilist(message)

            images = [Image.open(s).convert("RGB") for s in images]
            prompt = self.system_prompt + "USER: " + content + " ASSISTANT: "
            inputs = self.processor(prompt, images, return_tensors='pt').to(
                "cuda", dtype=torch.float16
            )
            output_ids = self.model.generate(
                    **inputs,
                    **self.kwargs
            )

            full_response = self.processor.batch_decode(output_ids.sequences,skip_special_tokens=True)[
                0
            ].strip()
            answer = full_response.split("ASSISTANT:")[1].strip()  # 只保留ASSISTANT部分

            return answer

    # def chat_inner(self, message, dataset=None):
    #     # 结构化对话历史
    #     conversation = []
    #     images = []
        
    #     # 解析多轮对话
    #     for msg in message:
    #         role = msg["role"]
    #         content_segments = []
            
    #         # 解析每条消息的内容（支持图文交替）
    #         for segment in msg["content"]:
    #             if segment["type"] == "text":
    #                 content_segments.append({"type": "text", "text": segment["value"]})
    #             elif segment["type"] == "image":
    #                 content_segments.append({"type": "image"})
    #                 # 加载并缓存图像
    #                 images.append(Image.open(segment["value"]).convert("RGB"))
            
    #         # 构建标准对话结构
    #         conversation.append({
    #             "role": "user" if role == "user" else "assistant",
    #             "content": content_segments
    #         })

    #     # 生成符合模板的prompt（自动添加系统消息和角色标识）
    #     prompt = self.processor.apply_chat_template(
    #         conversation,
    #         add_generation_prompt=True,  # 在最后添加ASSISTANT响应引导符
    #         tokenize=False
    #     )

    #     # 统一处理图像（支持跨多轮的图像输入）
    #     inputs = self.processor(
    #         text=prompt,
    #         images=images if images else None,  # 处理无图像的纯文本对话
    #         return_tensors="pt"
    #     ).to("cuda", torch.float16)

    #     # 配置生成参数（需与原始实现保持兼容）
    #     gen_kwargs = self.kwargs.copy()
    #     gen_kwargs.update({
    #         "eos_token_id": self.processor.tokenizer.eos_token_id,
    #         "pad_token_id": self.processor.tokenizer.pad_token_id
    #     })

    #     # 执行生成
    #     output = self.model.generate(**inputs, **gen_kwargs)
        
    #     # 解码和后处理
    #     answer = self.processor.decode(
    #         output[0],
    #         skip_special_tokens=True
    #     ).strip()
        
    #     # 移除可能的冗余内容（匹配原始实现逻辑）
    #     if self.stop_str:
    #         answer = answer.split(self.stop_str)[0].strip()
            
    #     return self.output_process(answer)
