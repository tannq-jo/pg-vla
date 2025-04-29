import argparse
import io
import itertools
import random
import string
from typing import Dict, List

import datasets
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from scripts.visualize import visualize_attention_over_prompt


def identify_sequence_structure(input_ids: torch.Tensor, processor, image_grid_size: int = 14) -> Dict[str, List[int]]:
    """
    Identify different parts of the input sequence structure for OpenVLA.

    Args:
        input_ids: The input token IDs [batch_size, sequence_length]
        processor: The OpenVLA processor/tokenizer
        image_grid_size: Size of the image patch grid (default 14 for 14x14 grid)

    Returns:
        Dictionary mapping each part of the sequence to its token indices
    """
    # Get the sequence (assuming batch size 1)
    sequence = input_ids[0].cpu().tolist()
    print(sequence)

    # In OpenVLA architecture:
    # - Token 0 is BOS (beginning of sequence)
    # - Next image_grid_size^2 tokens are image patches
    # - Final 7 tokens are action tokens
    # - Everything in between is the prompt

    n_image_patches = image_grid_size * image_grid_size  # 14x14 = 196 patches
    n_action_tokens = 7

    # Define the boundaries of each section
    token_indices = {
        "bos": [0],
        "image": list(range(1, 1 + n_image_patches)),
        "prompt": list(range(1 + n_image_patches, len(sequence) - n_action_tokens)),
        "action": list(range(len(sequence) - n_action_tokens, len(sequence))),
    }

    # Debug output to verify the segmentation
    print("\nSequence structure analysis:")
    for section_name, indices in token_indices.items():
        tokens = [sequence[idx] for idx in indices]
        decoded = processor.tokenizer.decode(tokens)
        token_count = len(indices)
        print(f"{section_name}: {token_count} tokens")
        print(f"  Indices: {indices[0]}...{indices[-1]}")
        print(f"  Decoded: {decoded[:100]}...")

    return token_indices


def test(head_fusion, random_text_prompt, random_input_image):
    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    config = OpenVLAConfig.from_pretrained("openvla/openvla-7b")

    vla = OpenVLAForActionPrediction.from_pretrained(
        "openvla/openvla-7b",
        config=config,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")

    if not random_input_image:
        dataset_names = [
            "fractal20220817_data",
            "stanford_hydra_dataset_converted_externally_to_rlds",
            "",
        ]
        data_per_dataset = {}
        for dataset_name in dataset_names:
            ds = datasets.load_dataset(
                "jxu124/OpenX-Embodiment",
                dataset_name,
                streaming=True,
                split="train",
                trust_remote_code=True,
            )  # IterDataset
            random_item = next(itertools.islice(ds, 10, 10 + 1))
            data_per_dataset[dataset_name] = random_item
        # Grab image input & format prompt
        # data = data_per_dataset["stanford_hydra_dataset_converted_externally_to_rlds"]
        data = data_per_dataset["fractal20220817_data"]
        image = Image.open(io.BytesIO(data["data.pickle"]["steps"][-1]["observation"]["image"]["bytes"]))
        image = image.resize((224, 224))

    last_attention = None
    while True:
        if not random_text_prompt:
            action_prompt = input("In: What action should the robot take to ")
            if action_prompt == "q":
                break
            prompt = f"In: What action should the robot take to {action_prompt }?\nOut:"
        else:
            prompt = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
        if random_input_image:
            random_array = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(random_array)

        # generate a random string

        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

        # outputs = vla(
        #    **inputs,
        #    return_dict=True,  # Ensure we get a dictionary output
        # )

        ## Convert pixel_values to patch tokens
        # pixel_values = inputs.pixel_values
        # patch_tokens = vla.vision_backbone(pixel_values)

        ## Print the shape of patch tokens
        # print("Patch tokens shape:", patch_tokens.shape)

        # logits = outputs.logits
        # probabilities = torch.softmax(logits, dim=-1)
        # predicted_token_ids = torch.argmax(probabilities, dim=-1)
        # decoded_output_text = processor.tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
        decoded_tokens = [
            processor.tokenizer.decode([token_id], skip_special_tokens=True) for token_id in inputs.input_ids[0, 1:]
        ]
        breakpoint()

        # print("Decoded output Text: ", decoded_output_text)
        # print("Decoded input Text: ", decoded_input_text)

        action, attention_data = vla.predict_action_with_attention(
            **inputs, unnorm_key="bridge_orig", do_sample=False, head_fusion=head_fusion
        )
        # breakpoint()
        # print(vla.get_ouput_embeddings().shape)

        print("Action: ", action)
        attention_rollout = attention_data["attention_rollout"].cpu()
        if last_attention is None:
            last_attention = attention_rollout

        # breakpoint()
        # print(attention_data["attention_rollout"].shape)
        # breakpoint()
        image_token_indices = list(range(16 * 16))
        sequence_length = attention_rollout.shape[-1]
        prompt_token_indices = list(range(16 * 16 + 1, 16 * 16 + inputs.input_ids.shape[-1]))
        print(prompt_token_indices)
        action_token_indices = list(range(sequence_length - 7, sequence_length))
        # print(image_token_indices)
        # print(action_token_indices)
        diff = 0
        for action_token_index in action_token_indices[:-1]:
            # diff += (
            #    attention_rollout[0, action_token_index, image_token_indices]
            #    - last_attention[0, action_token_index, image_token_indices]
            # ).sum()
            print(action_token_index)
            diff += (
                attention_rollout[0, action_token_index, image_token_indices]
                - attention_rollout[0, action_token_index + 1, image_token_indices]
            ).sum()

            # print(attention_rollout[0, action_token_index, image_token_indices])
            # break
        print("Diff: ", diff)
        last_attention = attention_rollout

        # visualize_attention(
        #    attention_rollout,
        #    image,
        #    image_token_indices,
        #    action_token_indices,
        #    title=action_prompt,
        # )

        visualize_attention_over_prompt(
            attention_rollout,
            prompt_token_indices,
            action_token_indices,
            x_ticks=decoded_tokens,
            title=action_prompt,
        )


if __name__ == "__main__":
    # Make an argparser that takes as input the possible prompt to openvla
    # and the image to be used
    args = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--head_fusion", type=str, choices=["mean", "max", "min"], default="mean")
    parser.add_argument("--set_random_text", type=bool, default=False)
    parser.add_argument("--set_random_image", type=bool, default=False)
    args = parser.parse_args()
    test(args.head_fusion, args.set_random_text, args.set_random_image)
