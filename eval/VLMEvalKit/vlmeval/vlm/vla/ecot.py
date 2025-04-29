from ..base import BaseModel
import pandas as pd
from PIL import Image
import string
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from vlmeval.dataset import DATASET_TYPE


ASCII_UPPERCASES = string.ascii_uppercase
PUNCTUATIONS = string.punctuation


class EmbodiedCoT(BaseModel):
    INTERLEAVE = False
    
    SYS_MSG = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "USER: {} ASSISTANT:"
    ) 
    
    COT_PROMPTS = [
        "Please think step by step."
    ]
    MCQ_PROMPTS = [
        (
            "Please select the correct option from the above choices based on the "
            "input image and question. The final output should only be one option, such as 'A'."
        ),
        "Answer with the option's letter from the given choices directly."
    ]
    OCR_PRE_PROMPTS = [
        (
            "Your task is to answer the question below. Give step by step reasoning."
            "\n\n"
            "Question:"
            "\n\n"
        ),
        (
            "Read the following question carefully, solve it step by step, and then output "
            "the final answer in the format of 'Answer: single number or single word or phrase'.\n\n"
        )
    ]
    OCR_POST_PROMPS = [
        "Answer the question using a single word or phrase.",
        "Carefully identify the text in the image and answer the question.",
        "Answer this question using the text in the image directly without any other context.",
        "Extract the text from the image intactly and answer the question concisely and clearly if possible.",
        "Give a very brief answer."
    ]
    YORN_PROMPTS = [
        "Yes or No?",
        "Please answer yes or no.",
        "Please answer yes or no as the final answer.",
        "Answer the question using a single word or phrase.",
        "Please answer yes or no directly, without any unnecessary explanation."
        "Please answer yes or no. Answer the question using a single word or phrase."
    ]

    USE_COT_FOR_OCR = True 

    DEFAULT_COT_PROMPT = \
        "Please think step by step."
    DEFAULT_OCR_PRE_PROMPT = \
        None
    DEFAULT_OCR_POST_PROMPT = \
        "Answer this question using the text in the image directly without any other context."
    DEFAULT_YORN_PROMPT = \
        "Respond with either 'Yes' or 'No' only, without any further explanation."
    

    
    def __init__(self, model_path: str):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device)

        self.data_keys = set()

    def use_custom_prompt(self, dataset: str) -> bool:
        return True
    
    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        dataset_type = DATASET_TYPE(dataset, default=None)

        self.data_keys |= set(line.keys())
        
        if dataset_type == "Y/N":
            return self._build_yorn_prompt(line, dataset)
        elif dataset_type == "VQA":
            return self._build_vqa_prompt(line, dataset)
        elif dataset_type == "MCQ":
            return self._build_mcq_prompt(line, dataset)

        ValueError(f"Unsupported dataset: {dataset}")

    def _build_mcq_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        msgs = []
        
        tgt_path = self.dump_image(line, dataset)
        if isinstance(tgt_path, list): # No interleave
            msgs.append(dict(type="image", value=tgt_path[0]))
        else:
            msgs.append(dict(type="image", value=tgt_path))
        
        question = line["question"].strip()
        question = question.replace(
            "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: ",
            ""
        )
        prompt = f"\nQuestion: {question}"
        
        if dataset == "MMStar":
            options = {
                cand: line[cand].strip()
                for cand in ASCII_UPPERCASES
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = "Options:\n"
            for k, v in options.items():
                options_prompt += f"{k}. {v}\n"

            if options:
                prompt += "\n" + options_prompt
                prompt += f"Note: {self.DEFAULT_MCQ_PROMPT}\n"
            else:
                prompt += "Note: Carefully read the following question Answer the question directly."

        
        msgs.append(dict(type="text", value=prompt))
        msgs.append(dict(type="text", value="mcq"))        
        
        return msgs


    def _build_vqa_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        msgs = []

        tgt_path = self.dump_image(line, dataset)
        if isinstance(tgt_path, list): # No interleave
            msgs.append(dict(type="image", value=tgt_path[0]))
        else:
            msgs.append(dict(type="image", value=tgt_path))

        question = line["question"].strip()

        if dataset == "OCRBench":
            self.processor.min_pixels = 10 * 10 * 28 * 28
            self.processor.max_pixels = 1280 * 28 * 28
            
            options = {
                cand: line[cand].strip()
                for cand in ASCII_UPPERCASES
                if cand in line and not pd.isna(line[cand])
            }

            options_prompt = ""
            for k, v in options.items():
                options_prompt += f"{k}. {v}\n"
 
            if self.DEFAULT_OCR_PRE_PROMPT is not None:
                prompt = self.DEFAULT_OCR_PRE_PROMPT
                punc = "\n"
            else:
                prompt = ""
                punc = " "
            
            prompt += question

            if options_prompt:
                prompt += "\n" + options_prompt
            
            if self.USE_COT_FOR_OCR and self.DEFAULT_OCR_PRE_PROMPT is None:
                prompt += punc + self.DEFAULT_COT_PROMPT
                punc = " "
            
            if self.DEFAULT_OCR_POST_PROMPT is not None:
                prompt += punc + self.DEFAULT_OCR_POST_PROMPT

            prompt = self.SYS_MSG.format(prompt) 

        msgs.append(dict(type="text", value=prompt))

        return msgs
    
    def _build_yorn_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        msgs = []
        
        tgt_path = self.dump_image(line, dataset)
        if isinstance(tgt_path, list): # No interleave
            msgs.append(dict(type="image", value=tgt_path[0]))
        else:
            msgs.append(dict(type="image", value=tgt_path))

        prompt = line["question"].strip() + " " + self.DEFAULT_YORN_PROMPT
        prompt = self.SYS_MSG.format(prompt)
        msgs.append(dict(type="text", value=prompt))
        
        msgs.append(dict(type="text", value="yorn"))

        return msgs

    def generate_inner(self, msgs: list[dict[str, str]], dataset: str) -> str:
        # image = Image.open(msgs[0]["value"])

        # model_inputs = self.processor(msgs[1]["value"], image).to(self.vla.device, self.vla.dtype)
        # _, generated_ids = self.vla.predict_action(
        #     **model_inputs,
        #     max_new_tokens=1024
        # )

        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]

        # generated_text = self.processor.batch_decode(
        #     generated_ids, 
        #     skip_special_tokens=True
        # )[0]

        # # A little trick
        # if msgs[2]["value"] == "yorn":
        #     tmp = generated_text.lower()
        #     while tmp[-1] in PUNCTUATIONS:
        #         tmp = tmp[:-1]
            
        #     if tmp.startswith("yes") or tmp.endswith("yes"):
        #         generated_text = "Yes"
        #     elif tmp.startswith("no") or tmp.endswith("no"):
        #         generated_text = "No"

        # if msgs[2]["value"] == "mcq":
        #     tmp = generated_text.lower()
        #     while tmp[-1] in PUNCTUATIONS:
        #         tmp = tmp[:-1]
            
        #     if tmp.startwith("A") or tmp.endswith("A"):
        #         generated_text = "A"
        #     elif tmp.startwith("B") or tmp.endswith("B"):
        #         generated_text = "B"
        #     elif tmp.startwith("C") or tmp.endswith("C"):
        #         generated_text = "C"
        #     elif tmp.startwith("D") or tmp.endswith("D"):
        #         generated_text = "D"
        
        generated_text = "Yes"
        print(self.data_keys)

        return generated_text

        