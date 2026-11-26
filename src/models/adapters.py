"""Copy a trainable policy without importing optimizer, cursor, or RNG state."""


def restore_adapter(model, state):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    current = get_peft_model_state_dict(model, adapter_name="default")
    if current.keys() != state.keys() or any(current[k].shape != v.shape for k, v in state.items()):
        raise ValueError("Incompatible adapter keys or shapes")
    if set_peft_model_state_dict(model, state, adapter_name="default").unexpected_keys:
        raise ValueError("Unexpected policy weights")
