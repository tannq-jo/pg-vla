from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
import os
from PIL import Image
import torch
from transformers import (
    AutoModel, AutoProcessor,
    PreTrainedModel, ProcessorMixin
)
from typing import Dict, List, Tuple


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/tannq/working/vinrobotics/vr-vla/pg-vla/exp/models/spatialvla-4b-mix-224-pt")
    parser.add_argument("--image-path", type=str, default="/home/tannq/working/vinrobotics/vr-vla/pg-vla/exp/SpatialVLA/test/example.png")
    parser.add_argument("--prompt", type=str, default="What action should the robot take to pick the cup?")
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--save-dir", type=str, default="/home/tannq/working/vinrobotics/vr-vla/pg-vla/exp/SpatialVLA/test")
    args = parser.parse_args()
    return args


def load_processor_and_model(args: Namespace) -> Tuple[ProcessorMixin, PreTrainedModel]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        args.model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()

    return processor, model


def draw_heatmaps(
    args: Namespace,
    heatmaps: List[torch.Tensor],
    note: str
) -> None:
    img = cv2.imread(args.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    os.makedirs(args.save_dir, exist_ok=True)
        
    for i, heatmap in enumerate(heatmaps):
        heatmap = heatmap.cpu().numpy()
        
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_norm = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        img_overlayed = cv2.addWeighted(img, 1 - args.alpha, heatmap_color, args.alpha, 0)

        img_output = Image.fromarray(img_overlayed)
        
        save_path = os.path.join(args.save_dir, f"attention_{note}_{i}.jpg")
        img_output.save(save_path)

    return 


def prepare_attn_maps_for_visualization(
    attentions: torch.Tensor,
    input_ids: torch.Tensor,
    start_img_token_idx: int = 0,
    end_img_token_idx: int = 256
) -> List[torch.Tensor]:    
    fused_attentions = attentions.mean(dim=1) # head-fused, (B, num_img_tokens, Lp)

    print("Fused attentions' shape:", fused_attentions.shape)

    attn_maps = []

    for i, fused_attn in enumerate(fused_attentions): # (num_img_tokens, Lp)
        img_attn_scores = fused_attn.mean(dim=1) # gather attention scores of image tokens, (Li,)
        
        attn_map = img_attn_scores.reshape(16, 16)
        attn_map = torch.pow(attn_map, 0.9)

        attn_maps.append(attn_map)
        
    return attn_maps


@torch.no_grad()
def visualize_attentions(
    args: Namespace, 
    processor: ProcessorMixin, 
    model: PreTrainedModel,
    image: Image,
    model_inputs: Dict[str, torch.Tensor]
):
    print("\n\n")

    # str1 = "pick the cup"
    # str2 = "cup"

    # tokenized1 = processor.tokenizer(str1, return_tensors="pt")
    # tokenized2 = processor.tokenizer(str2, return_tensor)

    generated_ids, attentions = model.predict_action_with_attentions(
        model_inputs, 
        return_attentions=True
    )

    start_img_token_idx = 0
    end_img_token_idx = 256

    # extract attention scores between image and prompt
    start_prompt_token_idx = end_img_token_idx
    end_prompt_token_idx = model_inputs["input_ids"].shape[-1]
    prompt_attentions = attentions[0][-1].detach().float() # last layer, (B, num_heads, src_seq_len, tgt_seq_len)
    prompt_attentions = prompt_attentions[
        :, :, 
        start_img_token_idx:end_img_token_idx, 
        start_prompt_token_idx:end_prompt_token_idx
    ] # (B, num_heads, num_img_tokens, prompt_len)

    print("start_prompt_token_idx:", start_prompt_token_idx)
    print("end_prompt_token_idx:", end_prompt_token_idx)
    print("prompt_attentions.shape:", prompt_attentions.shape)

    # extract attention scores between image and (action + reason)
    action_reason_attentions = []
    for attn in attentions[1:]:
        attn_ = attn[-1].detach().float() # last layer, (B, num_heads, 1, tgt_seq_len)
        action_reason_attentions.append(attn_)
    action_reason_attentions = torch.cat(action_reason_attentions, dim=2) # (B, num_heads, action_reason_len, tgt_seq_len) 
    action_reason_attentions = action_reason_attentions.permute(0, 1, 3, 2) # (B, num_heads, tgt_seq_len, action_reason_len)
    action_reason_attentions = action_reason_attentions[:, :, start_img_token_idx:end_img_token_idx, :] # (B, num_heads, num_img_tokens, action_reason_len)

    print("action_reason_attentions.shape:", action_reason_attentions.shape)

    # draw heatmap between image and prompt
    prompt_attn_maps = prepare_attn_maps_for_visualization(
        prompt_attentions,
        input_ids=model_inputs["input_ids"],
        start_img_token_idx=0,
        end_img_token_idx=256,        
    )
    draw_heatmaps(args, prompt_attn_maps, "prompt")

    # draw heatmap between image and (reason + action)
    action_reason_attn_maps = prepare_attn_maps_for_visualization(
        action_reason_attentions,
        input_ids=model_inputs["input_ids"],
        start_img_token_idx=0,
        end_img_token_idx=256,
    )
    draw_heatmaps(args, action_reason_attn_maps, "action_reason")
    
    return


def main():
    args = parse_args()

    processor, model = load_processor_and_model(args)

    image = Image.open(args.image_path).convert("RGB")
    model_inputs = processor(images=[image], text=args.prompt, return_tensors="pt")

    if (args.mode == "visualize_attentions"):
        visualize_attentions(args, processor, model, image, model_inputs)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    

if __name__ == "__main__":
    main()