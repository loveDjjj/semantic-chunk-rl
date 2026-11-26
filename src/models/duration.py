"""Budget-conditioned categorical duration actions with identical behavior/replay forwards."""
from functools import lru_cache
import json
import torch
from scrl.plan import PlanLimits, instruction


@lru_cache(maxsize=128)
def reachable(slots, support):
    if slots < 0:raise ValueError("Negative segment count")
    if slots==0:return frozenset([0])
    return frozenset(v+n for v in reachable(slots-1,support) for n in support)


def choices(remaining,slots,limits=PlanLimits()):
    if not 1<=slots<=limits.max_segments or not 1<=remaining<=limits.max_total_chunks:
        raise ValueError("Invalid duration decision")
    return tuple(n for n in limits.allowed_chunks if remaining-n in reachable(slots-1,limits.allowed_chunks))


def distribution(model,tok,prompt,captions,assigned,total):
    """Frozen captions are supplied externally; only duration choices are policy actions."""
    i=len(assigned);limits=PlanLimits()
    if not 1<=len(captions)<=limits.max_segments or i>=len(captions):
        raise ValueError("Invalid segment count")
    options=choices(total-sum(assigned),len(captions)-i)
    if not options:raise ValueError("Budget not achievable")
    if any(n not in limits.allowed_chunks for n in assigned):raise ValueError("Unsupported length")
    device=next(model.parameters()).device
    if len(options)==1:return options,torch.zeros(1,device=device)
    text=tok.apply_chat_template([dict(role="user",content=instruction(prompt))],
                                  tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefix="["
    for k in range(i+1):
        if k:prefix+=","
        prefix+=json.dumps(dict(step_index=k,prompt=captions[k]),separators=(",",":"),ensure_ascii=False)[:-1]+',"num_chunks":'
        if k<i:prefix+=str(assigned[k])+"}"
    ids=tok(text+prefix,return_tensors="pt",add_special_tokens=False).input_ids.to(device)
    tokens=[tok.encode(str(n),add_special_tokens=False) for n in options]
    if not all(len(t)==1 for t in tokens):raise ValueError("Expected single-token duration")
    logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1).logits[0,-1]
    return options,logits[[t[0] for t in tokens]].float().log_softmax(-1)
