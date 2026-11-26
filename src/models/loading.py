"""Local model loading and the generation prefix for SFT and RL planners."""


def load_planner(model_path, *, device, for_sft=False):
    import torch
    from transformers import AutoTokenizer

    tokenizer_options = {"padding_side": "left"} if for_sft else {}
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, **tokenizer_options)
    if for_sft:
        # SFT is 4B-only and leaves attention selection at the model default.
        from transformers import Qwen3_5ForConditionalGeneration as loader
        model_options = {"dtype": torch.bfloat16}
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        if config.model_type == "qwen3_5":
            from transformers import Qwen3_5ForConditionalGeneration as loader
        else:
            from transformers import AutoModelForCausalLM as loader
        model_options = {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}

    model = loader.from_pretrained(model_path, local_files_only=True, **model_options)
    return tokenizer, model.to(device)


def generation_prefix(tokenizer, user_message):
    return tokenizer.apply_chat_template(
        [user_message], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
