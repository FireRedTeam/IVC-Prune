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
                 ivcp_config = None,
                 **kwargs):

        from transformers import AutoProcessor, LlavaForConditionalGeneration
        self.system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.model_path = model_path


        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            ivcp_config = ivcp_config, 
            )

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            )

        model = model.eval()
        self.model = model.cuda()

        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
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
            answer = full_response.split("ASSISTANT:")[1].strip()  

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
            answer = full_response.split("ASSISTANT:")[1].strip()  

            return answer
