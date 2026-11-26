"""SFT with paired, single-attempt held-out plan generation."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics

from scrl.plan import FinitePlan
from scrl.sft import recipe_args
from scrl.sft.data import epoch_batches, load_splits


def empty_output(run):
    if run.is_symlink() or (run.exists() and (not run.is_dir() or any(run.iterdir()))):
        raise ValueError(f"SFT output must be a new or empty directory: {run}")


def claim_output(run, receipt):
    run.mkdir(parents=True, exist_ok=True)
    empty_output(run)
    with (run / "training-inputs.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True, help="New or empty SFT output directory")
    p.add_argument("--model-path", type=Path, required=True, help="Local Qwen3.5-4B directory")
    p.add_argument("--dataset", type=Path, required=True, help="Reviewed teacher SFT export")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--manifest-sha256", required=True, help="Independently approved teacher manifest SHA-256")
    fields = {"seed": int, "learning_rate": float, "lora_rank": int, "lora_alpha": int,
              "validation_batch_size": int, "max_new_tokens": int}
    for key, typ in fields.items():
        p.add_argument("--"+key.replace("_", "-"), type=typ)
    args = recipe_args(p, argv, "train", fields)
    if args.epochs < 1:
        p.error("epochs must be positive")
    if not math.isfinite(args.learning_rate) or any(getattr(args,k)<=0 for k in fields if k!="seed"):
        p.error("learning rate, LoRA dimensions, batch size and token budget must be positive")
    run = args.run
    pin = args.manifest_sha256
    try:
        empty_output(run)
        train, val = load_splits(args.dataset, manifest_sha256=pin, required_format="reviewed-teacher-sft-v1")
        if not args.model_path.is_dir():
            raise ValueError(f"Missing local planner model: {args.model_path}")
    except (OSError, KeyError, ValueError) as error:
        p.error(str(error))
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from scrl.models.loading import generation_prefix, load_planner
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    dist.barrier(device_ids=[local])
    # Ranks share one output. A second independent job may pass the initial empty-directory check.
    reservation_error = [None]
    if rank == 0:
        try:
            claim_output(run, {"dataset_manifest_sha256": pin, "model_path": str(args.model_path.resolve()),
                               "settings": {key:getattr(args,key) for key in fields}, "epochs": args.epochs, "world_size": world})
        except (OSError, ValueError) as error:
            reservation_error[0] = str(error)
    failed = torch.tensor(int(reservation_error[0] is not None), device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        dist.destroy_process_group()
        raise RuntimeError(reservation_error[0] or "Rank zero could not reserve the SFT output")
    if rank == 0: print("Output reserved; loading planner", flush=True)
    torch.manual_seed(args.seed)
    tok, base = load_planner(args.model_path, device=torch.device("cuda", local), for_sft=True)

    def prefix(row):
        return generation_prefix(tok, row["messages"][0])

    model = get_peft_model(
        base, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                         target_modules=r".*language_model.*self_attn\.(q_proj|v_proj)")
    )

    def evaluate(arm):
        rows=val[rank::world];result=[];model.eval()
        for start in range(0,len(rows),args.validation_batch_size):
            batch=rows[start:start+args.validation_batch_size]
            inp=tok([prefix(r) for r in batch],padding=True,return_tensors="pt",add_special_tokens=False).to("cuda")
            with torch.no_grad():out=model.generate(**inp,max_new_tokens=args.max_new_tokens,do_sample=False,use_cache=True)
            for row,ids in zip(batch,out[:,inp.input_ids.shape[1]:]):
                text=tok.decode(ids,skip_special_tokens=True)
                try:
                    plan=FinitePlan.parse(text);lengths=[s.num_chunks for s in plan.segments]
                    err=None
                except (TypeError,KeyError,ValueError) as e:
                    lengths=[];err=str(e)
                result.append(dict(id=row["prompt_id"],response=text,lengths=lengths,error=err))
        (run/f"{arm}-{rank}.json").write_text(json.dumps(result))
    ddp=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False)
    model.train()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.learning_rate,weight_decay=0)
    def batches():
        for epoch in range(args.epochs):
            order=list(range(len(train)))
            random.Random(args.seed+epoch).shuffle(order)
            yield from epoch_batches(order,world)
    processed=0
    for step,batch in enumerate(batches()):
        ids=batch[rank::world];processed+=len(batch)
        seq=[];labs=[]
        for i in ids:
            r=train[i];a=tok.encode(prefix(r),add_special_tokens=False)
            b=tok.encode(r["messages"][1]["content"]+tok.eos_token,add_special_tokens=False)
            seq.append(torch.tensor(a+b));labs.append(torch.tensor([-100]*len(a)+b))
        assert seq, "Last global batch must contain at least one example per rank"
        pad=torch.nn.utils.rnn.pad_sequence
        x=pad(seq,batch_first=True,padding_value=tok.pad_token_id).cuda()
        labels=pad(labs,batch_first=True,padding_value=-100).cuda()
        mask=torch.arange(x.size(1),device="cuda")[None,:]<torch.tensor([s.numel() for s in seq],device="cuda")[:,None]
        n=labels.ne(-100).sum();total=n.clone();dist.all_reduce(total)
        optimizer.zero_grad(set_to_none=True)
        loss=ddp(input_ids=x,attention_mask=mask,labels=labels,use_cache=False).loss
        (loss*n*world/total).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if rank==0: print(step+1,processed,float(loss.detach()),flush=True)
    if rank==0:model.save_pretrained(run/"adapter")
    dist.barrier()
    with model.disable_adapter(): evaluate("base")
    evaluate("sft")
    dist.barrier()
    if rank==0:
        summary={}
        for arm in ("base","sft"):
            rows=[r for i in range(world) for r in json.loads((run/f"{arm}-{i}.json").read_bytes())]
            assert {r["id"] for r in rows}=={r["prompt_id"] for r in val} and len(rows)==len(val)
            ok=[r for r in rows if r["error"] is None]
            summary[arm]=dict(attempts=len(rows),valid=len(ok),format_rate=len(ok)/len(rows),
                varied_lengths=sum(len(set(r["lengths"]))>1 for r in ok),
                mean_segments=statistics.mean(len(r["lengths"]) for r in ok) if ok else None)
        (run/"comparison.json").write_text(json.dumps(dict(train=len(train),epochs=args.epochs,updates=step+1,
            validation=len(val),metrics=summary,dataset_manifest_sha256=pin,
            scope="Format and chunk use, not video quality"),indent=2))
    dist.destroy_process_group()


if __name__=="__main__":main()
