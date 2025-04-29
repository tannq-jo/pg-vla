from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor


def main():
    model_path = "/home/lmaotan/work/tannq/work/vr-vla/pg-vla/eval/models/spatialvla-4b-224-pt"
    image_path = "/home/lmaotan/work/tannq/work/vr-vla/pg-vla/eval/vis/SpatialVLA/test/example.png"

    image = Image.open(image_path).convert("RGB")
    prompt = "What action should the robot take to pick the cup?"
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_path,
        # attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True ,
        trust_remote_code=True, 
    ).eval().cuda()
    
    model_inputs = processor(images=[image], text=prompt, return_tensors="pt")
    generation_outputs = model.predict_action(model_inputs)

    actions = processor.decode_actions(generation_outputs, unnorm_key="bridge_orig/1.0.0")
    print(actions)

    return


if __name__ == "__main__":
    main()