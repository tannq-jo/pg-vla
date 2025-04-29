from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

def main():
    model_path = "tt1225/openvla-7b-devel"
    image_path = "/home/lmaotan/work/tannq/work/vr-vla/pg-vla/eval/vis/SpatialVLA/test/example.png"
    prompt = "In: What action should the robot take to pick the cup?\nOut:"

    image = Image.open(image_path).convert("RGB")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        # attn_implementation="sdpa",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to(device)

    model.config.output_attentions = True

    model_inputs = processor(prompt, image, return_tensors="pt").to(device, torch.bfloat16)
    action, generated_ids = model.predict_action(
        **model_inputs, 
        unnorm_key="bridge_orig", 
        do_sample=False,
        output_attentions=True,
        return_dict=True, 
    )


if __name__ == "__main__":
    main()
