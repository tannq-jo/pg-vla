import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
import json
import torch
import torch.distributed as dist
from train.dist_utils import init_dist
from train.monkey_patch import (
    replace_train_dataloader,
    replace_compute_loss,
    concat_pad_data_collator,
    replace_train_sampler,
    SaveProcessorCallback
)
import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    set_seed,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity,
)
from data.dataset import build_datasets
from model import (
    SpatialVLAConfig,
    SpatialVLAForConditionalGeneration,
    SpatialVLAProcessor,
    SpatialActionTokenizer,
)
replace_train_dataloader()
replace_compute_loss()
replace_train_sampler()

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(default=None,
        metadata={"help": "Path to pretrained model or identifier for resume training."},
    )
    freeze_llm_embed: bool = field(
        default=True, metadata={"help": "Set to True to freeze the LLM embeddings."},
    )
    freeze_vision_tower: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the vision backbone of the model."},
    )
    lora: int = field(
        default=0,
        metadata={"help": "Set the LoRA adapter rank for the LLM. Default is 0."},
    )
    lora_alpha: int = field(
        default=8,
        metadata={"help": "Set the LoRA adapter rank for the LLM. Default is 0."},
    )
    lora_target: Optional[str] = field(
        default="linear",
        metadata={"help": "Set the LoRA adapter rank for the LLM. Default is linear."},
    )
    modules_to_save: Optional[str] = field(
        default=None,
        metadata={"help": "Set the LoRA adapter rank for the LLM. Default is none."},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to use gradient checkpointing."},
    )
    flash_attn: bool = field(
        default=True,
        metadata={"help": "Set to True to use Flash Attention 2.0."},
    )
    adapt_emb: Optional[str] = field(
        default=None,
        metadata={"help": "Set to True to adapt the spatial embeddings with new gaussian config."},
    )
    adpt_feature: bool = field(
        default=False,
        metadata={"help": "Set to True to adapt the feature embeddings."},
    )
    min_sigma: float = field(
        default=0.0,
        metadata={"help": "Set the minimum sigma for creating action grids."},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_root_dir: Optional[str] = field(
        default="datasets/open-x-embodiment",
        metadata={"help": "The root directory of the dataset. Default is `data`."},
    )
    data_mix: Optional[str] = field(
        default="bridge",
        metadata={"help": "The name of the dataset mixture. Default is `bridge`."},
    )
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length after tokenization. "},
    )
    shuffle_buffer_size: Optional[int] = field(
        default=1000_000,
        metadata={"help": "The shuffle buffer size for the dataset. Default is 1000000."},
    )
    tsfm_thread_muti: Optional[int] = field(
        default=1,
        metadata={"help": "The threads number of rlds transfom. Default is 1."},
    )
    read_thread_muti: Optional[int] = field(
        default=1,
        metadata={"help": "The threads number of rlds reader. Default is 1."},
    )
    obs_backward_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of backward steps in observation. 0 indicates current"},
    )
    obs_backward_delta: Optional[int] = field(
        default=1, metadata={"help": "Backward delta in observation."}
    )
    action_forward_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of forward steps in action. 0 indicates current"},
    )
    fix_raw_length: Optional[int] = field(
        default=None, metadata={"help": "fix the iterable dataset iter length."}
    )
    use_raw_dataloader: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use raw dataloader"}
    )

def main():
    launcher = os.environ.get("LAUNCHER", "slurm")
    init_dist(launcher=launcher, backend="nccl")
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log: transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        ckpt_files = list(filter(lambda x: x.startswith("checkpoint"), os.listdir(training_args.output_dir)))
        if last_checkpoint is None and len(ckpt_files) > 0:
            ckpt_files = list(filter(lambda x: x.startswith("checkpoint"), os.listdir(training_args.output_dir)))
        if last_checkpoint is None and len(ckpt_files) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    # 1. initializing models and load tokenizer
    _processor = SpatialVLAProcessor.from_pretrained(model_args.model_name_or_path, local_files_only=True)
    tokenizer = _processor.tokenizer
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    
    logger.info("Loading SpatialVLA Model...")
    config = SpatialVLAConfig.from_pretrained(model_args.model_name_or_path, torch_dtype=torch_dtype, local_files_only=True)
    model = SpatialVLAForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        local_files_only=True
    )
    if model_args.flash_attn:
        model.language_model.config._attn_implementation = model.config.text_config._attn_implementation_internal = "flash_attention_2"
        model.vision_tower.config._attn_implementation = model.config.vision_config._attn_implementation_internal = "flash_attention_2"

    # 2. build datasets
    train_dataset, eval_dataset = build_datasets(
        data_args,
        training_args.output_dir,
        vla_processor=None,
    )

    # 3. build action tokenizer from current project
    action_tokenizer = SpatialActionTokenizer(
        tokenizer,
        num_bins=_processor.action_config["num_bins"],
        bin_policy=_processor.action_tokenizer.bin_policy,
        use_spherical=_processor.action_config["use_spherical"],
        min_sigma=_processor.action_config.get("min_sigma", 0.0),
    )
    
    if model_args.adapt_emb and config.use_spatial_token:
        logger.info(f"adapt spatial embeddings with guassian distribution {model_args.adapt_emb}")
        gs_params = json.load(open(model_args.adapt_emb))
        action_tokenizer.spatial_embedding_adaption(gs_params, model.spatial_embed_tokens, model_args.min_sigma, model_args.adpt_feature)
        logger.info(f"new adaptation embedding {model.spatial_embed_tokens.weight.data}")

        if model_args.adpt_feature:
            model_args.lora_target="linear"
            model_args.modules_to_save="spatial_embed_tokens"
            logger.info(f"reset lora_target to {model_args.lora_target} and modules_to_save {model_args.modules_to_save}")

    # overwrite attributes
    model.action_token_begin_idx = model.config.action_token_begin_idx = action_tokenizer.action_token_begin_idx
    model.vision_tower.gradient_checkpointing = True

    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()
    
    # set freeze params
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_llm_embed:
        model.language_model.model.embed_tokens.weight.requires_grad = False

    if model_args.freeze_vision_tower:
        model.vision_tower = model.vision_tower.eval()
        _freeze_params(model.vision_tower)

    model.vision_zoe_model = model.vision_zoe_model.eval()
    _freeze_params(model.vision_zoe_model)

    if model_args.lora:
        # peft https://github.com/huggingface/peft/blob/c1fe8105a5a4a612a6178699e1def5c66c2638d2/src/peft/tuners/tuners_utils.py#L1027
        if model_args.lora_target == "linear":
            target_modules=[
                "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", # com
                "fc1", "fc2", "out_proj", # siglip
                "linear", # projector
                "position_embedding_head.0", "position_embedding_head.3" # ego3d
            ]
        elif model_args.lora_target == "linear+emb":
            target_modules=[
                "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", # com
                "fc1", "fc2", "out_proj", # siglip
                "linear", # projector
                "position_embedding_head.0", "position_embedding_head.3", # ego3d
                "spatial_embed_tokens",
            ]
        elif model_args.lora_target == "linear+emb+h":
            target_modules=[
                "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", "lm_head", # com
                "fc1", "fc2", "out_proj", # siglip
                "linear", # projector
                "position_embedding_head.0", "position_embedding_head.3", # ego3d
                "spatial_embed_tokens",
            ]
        else:
            raise ValueError(f"don't support lora targets {model_args.lora_target}")
        
        # modules_to_save: https://github.com/huggingface/peft/issues/334#issuecomment-1786449397
        modules_to_save = model_args.modules_to_save.split("+") if model_args.modules_to_save else []
        lora_config = LoraConfig(
            r=model_args.lora,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            init_lora_weights="gaussian",
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"use Lora ... with {model_args.lora_target} and modules {modules_to_save} ...")
        model.print_trainable_parameters()

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad: logger.info(name)

    set_seed(training_args.seed)
    SpatialVLAConfig.register_for_auto_class() # register for auto save and map
    SpatialVLAForConditionalGeneration.register_for_auto_class()
    SpatialVLAProcessor.register_for_auto_class()

    # build processor
    statistic = train_dataset.ds_stats_pc
    _processor.statistics.update(statistic)
    processor = SpatialVLAProcessor(
        image_processor=_processor.image_processor,
        tokenizer=tokenizer,
        statistics=_processor.statistics,
        bin_policy=action_tokenizer.bin_policy,
        intrinsic_config=_processor.intrinsic_config,
        action_config=_processor.action_config,
        num_obs_steps=data_args.obs_backward_steps + 1,
        obs_delta=data_args.obs_backward_delta,
        action_chunk_size=data_args.action_forward_steps + 1,
    )

    model.action_tokenizer = action_tokenizer
    train_dataset.vla_processor = processor

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=concat_pad_data_collator,
        callbacks=[SaveProcessorCallback(processor=processor)],
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

if __name__ == "__main__":
    main()
