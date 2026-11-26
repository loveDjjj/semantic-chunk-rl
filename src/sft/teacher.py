"""从独立训练提示词生成待审核的分段草稿；不会直接生成训练标签。"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from scrl.models.responses import (TEACHER_TEMPLATE_VERSION, FatalTeacherError, InvalidTeacherPlanError,
                                  RetryableTeacherError, TeacherRefusalError,
                                  relay_url, request_teacher_plan, teacher_request_sha256,
                                  teacher_request_template, teacher_template_sha256)
from scrl.plan import FinitePlan, digest
from scrl.sft import recipe_args
from scrl.sft.data import corpus_policy, nonempty, prompt_key


def atomic_text(path, text):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def corpus(path, blocked):
    raw = path.read_bytes()
    rows = {}
    for line, data in enumerate(raw.splitlines(), 1):
        row = json.loads(data)
        prompt = nonempty(row["prompt"], "prompt")
        group = nonempty(row["group"], "group")
        key = prompt_key(prompt)
        if key in rows or key in blocked or prompt_key(group) in blocked:
            raise ValueError(f"第 {line} 行提示词重复或与留出集重叠")
        rows[key] = (prompt, group)
    if not rows:
        raise ValueError("训练提示词不能为空")
    return rows, hashlib.sha256(raw).hexdigest()


def generate(args):
    blocked, exclusions, allowed, included = corpus_policy(args.exclude_metadata,
                                                          args.include_official_metadata)
    rows, source = corpus(args.prompts, blocked)
    model = nonempty(args.model, "model")
    effort = nonempty(args.effort, "effort")
    manifest = dict(format="teacher-drafts-v1", prompts_sha256=source,
                    exclusions_sha256=exclusions, model=model, effort=effort,
                    template_version=TEACHER_TEMPLATE_VERSION,
                    template_sha256=teacher_template_sha256(effort), count=len(rows))
    if included:
        manifest.update(official_training_metadata_sha256=included,
                        official_training_prompt_count=len(rows.keys() & allowed))
    if args.dry_run:
        print(json.dumps(manifest))
        return
    url = relay_url(args.base_url)
    manifest["relay_sha256"] = digest(url)
    root = args.run
    root.mkdir(parents=True, exist_ok=True)
    with (root/".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pin = root/"manifest.json"
        if pin.exists():
            if json.loads(pin.read_bytes()) != manifest:
                raise ValueError("输入、教师或协议已变，请使用新的运行目录")
        else:
            if any(p.name != ".lock" for p in root.iterdir()):
                raise ValueError("拒绝接管已有文件的目录")
            atomic_text(pin, json.dumps(manifest, indent=2) + "\n")
        template_path = root/"teacher_request_template.json"
        template = teacher_request_template(effort)
        if template_path.exists():
            if json.loads(template_path.read_bytes()) != template:
                raise ValueError("Saved teacher request template changed")
        else:
            atomic_text(template_path, json.dumps(template, indent=2, ensure_ascii=False) + "\n")
        cache = root/"cache"
        cache.mkdir(exist_ok=True)
        pending, failed = [], {}
        refusals = root/"refusals"
        def read_cached(key):
            item = json.loads((cache/(key+".json")).read_bytes())
            prompt, group = rows[key]
            draft = item["draft"]
            if (digest(draft) != item["sha256"] or draft.get("effort") != effort
                    or draft["request_sha256"] != teacher_request_sha256(prompt, model, effort)
                    or draft["prompt"] != prompt or draft["group"] != group or draft["teacher_model"] != model
                    or draft["template_sha256"] != manifest["template_sha256"]
                    or draft.get("official_training_metadata_sha256", []) != included
                    or draft["approved"] is not False or draft["reviewer"] is not None):
                raise ValueError("缓存身份不匹配")
            FinitePlan.parse(json.dumps(draft["plan"]))
            return draft
        for key in rows:
            if (cache/(key+".json")).exists():
                read_cached(key)
            elif (refusals/key).is_file():
                failed[key] = "refused"
            else:
                pending.append(key)
        if failed and not args.continue_invalid:
            raise TeacherRefusalError("Existing prompt refusals; refusing to resubmit them")
        if pending and not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("缺少 OPENAI_API_KEY")
        def one(key):
            prompt, group = rows[key]
            for attempt in range(args.attempts):
                try:
                    plan, receipt = request_teacher_plan(prompt, model, url, effort=effort)
                    break
                except RetryableTeacherError:
                    if attempt + 1 == args.attempts:
                        raise
                    time.sleep(min(8, 2**attempt))
            draft = dict(prompt=prompt, group=group, teacher_model=model, plan=plan.to_output(),
                         reviewer=None, approved=False, **receipt)
            if included:
                draft["official_training_metadata_sha256"] = included
            atomic_text(cache/(key+".json"), json.dumps(dict(draft=draft, sha256=digest(draft))))
        # Only fill newly available slots after checking every completed request.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = iter(pending)
            live = {}
            while True:
                while len(live) < args.workers and (key := next(pending, None)) is not None:
                    live[pool.submit(one, key)] = key
                if not live:
                    break
                for future in wait(live, return_when=FIRST_COMPLETED).done:
                    key = live.pop(future)
                    try:
                        future.result()
                    except (TeacherRefusalError, InvalidTeacherPlanError) as error:
                        if isinstance(error, TeacherRefusalError):
                            refusals.mkdir(exist_ok=True)
                            atomic_text(refusals/key, "refused\n")
                            failed[key] = "refused"
                        else:
                            failed[key] = "invalid_plan"
                        if not args.continue_invalid:
                            raise
        if failed:
            atomic_text(root/"failures.jsonl", "".join(json.dumps({"prompt_id": k, "reason": failed[k]})+"\n"
                                                       for k in rows if k in failed))
            atomic_text(root/"review.partial.jsonl", "".join(json.dumps(read_cached(k), ensure_ascii=False)+"\n"
                                                            for k in rows if k not in failed))
            raise ValueError(f"Incomplete teacher batch: {len(failed)}/{len(rows)} without validated drafts")
        result = "".join(json.dumps(read_cached(k), ensure_ascii=False)+"\n" for k in rows)
        target = root/"review.jsonl"
        if target.exists() and target.read_text(encoding="utf-8") != result:
            raise ValueError("已有审核文件被修改，不覆盖")
        if not target.exists():
            atomic_text(target, result)
        if (root/"failures.jsonl").exists():
            atomic_text(root/"failures.jsonl", "")
            (root/"review.partial.jsonl").unlink(missing_ok=True)
        print(f"待审核 {len(rows)} 条：{target}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--exclude-metadata", type=Path, action="append", default=[])
    p.add_argument("--include-official-metadata", type=Path, action="append", default=[])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model")
    p.add_argument("--effort", help="Teacher reasoning effort, e.g. low")
    p.add_argument("--base-url")
    p.add_argument("--workers", type=int)
    p.add_argument("--attempts", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-invalid", action="store_true",
                   help="Skip refused/invalid drafts, export partial results and exit nonzero if incomplete")
    args = recipe_args(p, argv, "teacher", {"model": str, "effort": str, "workers": int, "attempts": int})
    if args.workers < 1 or args.attempts < 1:
        p.error("workers 和 attempts 必须为正数")
    try:
        generate(args)
    except (OSError, ValueError, KeyError, TypeError, FatalTeacherError, RetryableTeacherError) as error:
        p.exit(2, f"{p.prog}: {error}\n")


if __name__ == "__main__":
    main()
