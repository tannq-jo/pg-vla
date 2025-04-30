from argparse import ArgumentParser, Namespace
from PIL import Image
import torch
from transformers import (
    AutoModel, AutoProcessor,
    PreTrainedModel, ProcessorMixin
)
from typing import Dict


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/tannq/working/vinrobotics/vr-vla/pg-vla/exp/models/spatialvla-4b-mix-224-pt")
    parser.add_argument("--image-path", type=str, default="/home/tannq/working/vinrobotics/vr-vla/pg-vla/exp/SpatialVLA/test/example.png")
    parser.add_argument("--prompt", type=str, default="What action should the robot take to pick the cup?")
    parser.add_argument("--mode", type=str, default="")
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


@torch.no_grad()
def visualize_attentions(
    args: Namespace, 
    processor: ProcessorMixin, 
    model: PreTrainedModel,
    image: Image,
    model_inputs: Dict[str, torch.Tensor]
):
    for k, v in model_inputs.items():
        print(f"{k}: {type(v)}")

        if isinstance(v, list):
            for i in v:
                print(f"\t- {i.shape}")    
        else:
            print(f"\t- {v.shape}")
    
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