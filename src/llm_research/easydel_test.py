import easydel as ed
from transformers import AutoTokenizer
from datasets import load_dataset
import jax.numpy as jnp
from jax import lax

# Load model with full configuration options
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
max_length = 2048

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_id,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    precision=lax.Precision.DEFAULT,
    backend=ed.EasyDeLBackends.GPU,
    platform=ed.EasyDeLPlatforms.TRITON,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    # (DP, FSDP, EP, TP, SP) - Full TP
    config_kwargs=ed.EasyDeLBaseConfigDict(
        freq_max_position_embeddings=max_length,
        mask_max_position_embeddings=max_length,
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
        attn_dtype=jnp.bfloat16,
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    ),
    partition_axis=ed.PartitionAxis(),  # Default partitioning
)

# DPO is used to align models with human preferences (e.g., Llama 3, GPT-4)
trainer = ed.DPOTrainer(
    model=model,
    arguments=ed.DPOConfig(
        beta=0.1,  # KL penalty coefficient
        loss_type="sigmoid",  # or "ipo", "hinge"
        max_length=512,
        max_prompt_length=256,
        max_completion_length=256,
        total_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=5e-7,
        scheduler=ed.EasyDeLSchedulers.LINEAR,
        num_train_epochs=1,
        ref_model_sync_steps=128,
        precompute_ref_log_probs=False,
        disable_dropout=True,
        save_steps=1000,
        report_steps=20,
    ),
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
    processing_class=AutoTokenizer.from_pretrained(model_id),
)

trainer.train()