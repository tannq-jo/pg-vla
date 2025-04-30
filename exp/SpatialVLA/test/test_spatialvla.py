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
    parser.add_argument("--mode", type=str, default="")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--save-dir", type=str, default="/home/tannq/working/vinrobotics/vr-vla/pg-vla/exp/SpatialVLA/test")
    args = parser.parse_args()
    return args


def load_processor_and_model(args: Namespace):
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


def draw_heatmap(
    args: Namespace,
    heatmaps: List[torch.Tensor],
):
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
        save_path = os.path.join(args.save_dir, f"attention_{i}.jpg")
        img_output.save(save_path)

    return 


def prepare_attn_maps_for_visualization(
    attentions: Tuple[Tuple[torch.Tensor]],
    input_ids: torch.Tensor,
    start_img_token_index: int = 0,
    end_img_token_index: int = 256,
):
    attentions_ = torch.stack(attentions[0], dim=0) # (Nl, B, Nh, Ls, Lt)
    attentions_ = attentions_.permute(1, 0, 2, 3, 4) # (B, Nl, Nh, Ls, Lt)
    attentions_ = attentions_.detach().float()

    prompt_len = input_ids.shape[-1] - end_img_token_index
    start_prompt_token_index = end_img_token_index
    end_prompt_token_index = start_prompt_token_index + prompt_len

    # fused_attentions = attentions_.mean(dim=1) # layer-fursed, (B, Nh, Ls, Lt)
    fused_attentions = attentions_[:, -1, :, :, :] # last layer, (B, Nh, Ls, Lt)
    fused_attentions = fused_attentions[:, :, :, start_prompt_token_index : end_prompt_token_index]
    fused_attentions = fused_attentions.mean(dim=1) # head-fused, (B, Ls, Lt)

    attn_maps = []

    for fused_attn in fused_attentions: # (Ls, Lt)
        src_attn_scores = fused_attn.mean(dim=1) # gather attention scores of source tokens, (Ls,)
        img_attn_scores = src_attn_scores[start_img_token_index:end_img_token_index]

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
    # print("\n\n")

    # for k, v in model_inputs.items():
    #     print(f"{k}: {type(v)}")

    #     if isinstance(v, list):
    #         for i in v:
    #             print(f"\t- {i.shape}")    
    #     else:
    #         print(f"\t- {v.shape}")

    # print("Input IDs:", model_inputs["input_ids"])
    # print("Special tokens:", processor.tokenizer.convert_ids_to_tokens([257152, 2, 108]))

    generated_ids, attentions = model.predict_action_with_attentions(model_inputs, True)

    attn_maps = prepare_attn_maps_for_visualization(
        attentions,
        model_inputs["input_ids"],
        start_img_token_index=0,
        end_img_token_index=256,        
    )

    draw_heatmap(args, attn_maps)
    
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