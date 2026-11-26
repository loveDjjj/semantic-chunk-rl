"""Transformers reference implementation for finite planning and focal updates.

The same token/mask payload is the interface to distributed planner trainers.
"""

from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path
import string
import torch

from scrl.plan import DecisionTokens, FinitePlan, PlanLimits, Segment, digest, instruction
from scrl.models.loading import generation_prefix, load_planner


class FinitePlanGrammar:
    """Constrain JSON syntax while preserving sampled captions and group lengths."""

    def __init__(self, tokenizer, context_length, limits=PlanLimits()):
        self.tokenizer, self.context_length, self.limits = tokenizer, context_length, limits
        if any(n > 9 for n in limits.allowed_chunks):
            raise ValueError("This grammar currently supports single-digit chunk choices")
        vocab = tokenizer.get_vocab()
        self.size = max(vocab.values()) + 1
        self.pieces = tokenizer.batch_decode([[i] for i in range(self.size)],
            skip_special_tokens=False, clean_up_tokenization_spaces=False)
        self.special = set(tokenizer.all_special_ids)
        self.constraints, self.fields, self.mask_cache = {}, {}, {}
        self.device_support = {}
        self.eos_support = [tokenizer.eos_token_id]
        self.by_first_character = {}
        for token, piece in enumerate(self.pieces):
            if token not in self.special and piece:
                self.by_first_character.setdefault(piece[0], []).append(token)
        self.numbers = {n: tokenizer.encode(str(n), add_special_tokens=False)
                        for n in limits.allowed_chunks}
        if any(len(ids) != 1 for ids in self.numbers.values()):
            raise ValueError("Chunk values must each be a single token")

    @staticmethod
    def initial():
        return {"mode": "literal", "literal": '[{"step_index":0,"prompt":"',
                "after": "caption", "nonempty": False, "hex": 0, "lengths": [],
                "literal_string": False}

    def consume(self, state, text, *, numbers=True, events=None, boundaries=None):
        for char in text:
            mode = state["mode"]
            if mode == "literal":
                if char in " \t\r\n" and not state["literal_string"]:
                    continue
                if not state["literal"] or char != state["literal"][0]:
                    return False
                state["literal"] = state["literal"][1:]
                if char == '"':
                    state["literal_string"] = not state["literal_string"]
                if not state["literal"]:
                    state["mode"] = state["after"]
                    if state["mode"] == "caption":
                        state["nonempty"] = False
                        state["literal_string"] = False
            elif mode == "caption":
                if events is not None:
                    events.add(("caption", len(state["lengths"])))
                if char == '"':
                    if not state["nonempty"]:
                        return False
                    state.update(mode="literal", literal=',"num_chunks":', after="number", literal_string=False)
                elif char == "\\":
                    state["mode"] = "escape"
                elif ord(char) < 32:
                    return False
                else:
                    state["nonempty"] |= not char.isspace()
            elif mode == "escape":
                if events is not None:
                    events.add(("caption", len(state["lengths"])))
                if char == "u":
                    state.update(mode="unicode", hex=4)
                elif char in '"\\/bfnrt':
                    state.update(mode="caption", nonempty=True)
                else:
                    return False
            elif mode == "unicode":
                if events is not None:
                    events.add(("caption", len(state["lengths"])))
                if char not in string.hexdigits:
                    return False
                state["hex"] -= 1
                if state["hex"] == 0:
                    state.update(mode="caption", nonempty=True)
            elif mode == "number":
                if char in " \t\r\n":
                    continue
                if not numbers or char not in {str(n) for n in self.limits.allowed_chunks}:
                    return False
                n = int(char)
                if sum(state["lengths"]) + n > self.limits.max_total_chunks:
                    return False
                if events is not None:
                    events.add(("duration", len(state["lengths"])))
                state["lengths"].append(n)
                state.update(mode="literal", literal="}", after="boundary", literal_string=False)
            elif mode == "boundary":
                if char in " \t\r\n":
                    continue
                if (boundaries is not None and char in "]," and
                    len(state["lengths"]) < self.limits.max_segments and
                    sum(state["lengths"]) + min(self.limits.allowed_chunks) <= self.limits.max_total_chunks):
                    boundaries.append(char)
                if char == "]":
                    state.update(mode="done")
                elif char == ",":
                    if (len(state["lengths"]) >= self.limits.max_segments or
                        sum(state["lengths"]) + min(self.limits.allowed_chunks) > self.limits.max_total_chunks):
                        return False
                    state.update(mode="literal", literal='{"step_index":' + str(len(state["lengths"])) + ',"prompt":"', after="caption", literal_string=False)
                else:
                    return False
            elif mode == "done" and char in " \t\r\n":
                continue
            else:
                return False
        return True

    def state(self, ids):
        text = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        state = self.initial()
        if not self.consume(state, text):
            raise ValueError("Generated text violates the finite-plan grammar")
        return state

    def allowed(self, state):
        mode = state["mode"]
        if mode == "done":
            return self.eos_support
        is_caption = mode in ("caption", "escape", "unicode")
        key = ((mode, state["nonempty"], state["hex"]) if is_caption else
               (mode, state["literal"], state["after"], state["literal_string"],
                len(state["lengths"]), sum(state["lengths"])))
        if key not in self.mask_cache:
            starts = None
            if mode == "literal":
                starts = set(state["literal"][:1])
                if not state["literal_string"]:
                    starts.update(" \t\r\n")
            elif mode == "number":
                starts = set(" \t\r\n" + "".join(str(n) for n in self.limits.allowed_chunks))
            elif mode == "boundary":
                starts = set(" \t\r\n],")
            candidates = (range(self.size) if starts is None else
                          (token for char in starts for token in self.by_first_character.get(char, [])))
            allowed = []
            for token in candidates:
                piece = self.pieces[token]
                events = set()
                if (token not in self.special and piece and
                    self.consume(copy.deepcopy(state), piece, numbers=not is_caption, events=events)
                    and len(events) <= 1):
                    allowed.append(token)
            self.mask_cache[key] = allowed
        return self.mask_cache[key]

    def decision_spans(self, token_ids, count, whole_positions=None):
        state = self.initial()
        positions = {}
        for position, token in enumerate(token_ids):
            token = int(token)
            if token == self.tokenizer.eos_token_id:
                if state["mode"] != "done":
                    raise ValueError("EOS occurred before the plan completed")
                break
            events = set()
            boundaries = [] if whole_positions is not None else None
            if not self.consume(state, self.pieces[token], events=events, boundaries=boundaries) or len(events) > 1:
                raise ValueError("Ambiguous or invalid token at a planning field boundary")
            if whole_positions is not None and (events or boundaries) and len(self.constraints[position]) > 1:
                whole_positions.append(position)
            for field in events:
                positions.setdefault(field, []).append(position)
        decisions = {}
        for index in range(count):
            caption = positions.get(("caption", index), [])
            duration = positions.get(("duration", index), [])
            if not caption or len(duration) != 1:
                raise ValueError("Missing planning decision tokens")
            decisions[index] = DecisionTokens((min(caption), max(caption) + 1),
                                               (duration[0], duration[0] + 1))
        return decisions

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1:
            raise ValueError("The reference grammar samples one plan at a time")
        state = self.state(input_ids[0, self.context_length:])
        allowed = self.allowed(state)
        if getattr(self, 'close_at', None) is not None and input_ids.shape[1] >= self.close_at:
            mode = state['mode']
            if mode != 'done':
                if mode == 'number':
                    wanted = [v[0] for v in self.numbers.values()]
                else:
                    char = {'escape':'"', 'unicode':'0', 'boundary':']',
                            'caption':'"' if state['nonempty'] else 'x'}.get(mode)
                    if mode == 'literal':
                        char = state['literal'][0]
                    wanted = self.tokenizer.encode(char, add_special_tokens=False)
                    if len(wanted) != 1 or self.pieces[wanted[0]] != char:
                        raise ValueError('Cannot encode grammar closure character')
                allowed = [i for i in wanted if i in allowed]
        if not allowed:
            raise ValueError("Plan has exhausted the generation resource budget")
        position = input_ids.shape[1] - self.context_length
        self.constraints[position] = allowed
        self.fields[position] = (len(state["lengths"]), state["mode"])
        result = torch.full_like(scores, -torch.inf)
        key = (id(allowed), scores.device)
        if key not in self.device_support:
            # Keep the source list alive: Python can otherwise reuse an old EOS
            # list's id for a later support, turning a caption mask into EOS.
            self.device_support[key] = (allowed, torch.tensor(allowed, device=scores.device, dtype=torch.long))
        support = self.device_support[key][1]
        result.index_copy_(1, support, scores.index_select(1, support))
        return result


def token_log_probs(model, context, response, constraints):
    ids = torch.cat((context, response), dim=1)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    scores = logits[:, context.shape[1] - 1:-1]
    parts, support_cache = [], {}
    for begin in range(0, response.shape[1], 64):
        chunk = scores[:, begin:begin + 64].float()
        mask = torch.zeros_like(chunk, dtype=torch.bool)
        for local in range(chunk.shape[1]):
            allowed = constraints.get(begin + local)
            if allowed is None:
                raise ValueError("Missing behavior sampling mask")
            # Caption positions share a large legal vocabulary. Convert that support
            # once, rather than copying a Python list of ~150k IDs for every token.
            key = id(allowed)
            if key not in support_cache:
                support_cache[key] = torch.tensor(allowed, device=scores.device, dtype=torch.long)
            mask[:, local, support_cache[key]] = True
        chunk = chunk.masked_fill(~mask, -torch.inf).log_softmax(-1)
        parts.append(chunk.gather(-1, response[:, begin:begin + 64, None]).squeeze(-1))
    return torch.cat(parts, dim=1)[0]


@dataclass
class FieldTrace:
    context: torch.Tensor
    response: torch.Tensor
    constraints: dict
    old_log_probs: torch.Tensor
    reference_log_probs: torch.Tensor


@dataclass
class CrossedPlannerSample:
    plan: FinitePlan
    caption_trace: FieldTrace | None
    duration_trace: FieldTrace | None
    policy_version: int


def crossed_terms(samples, advantages, targets, *, version, duration_weight):
    """Live field observations and their coefficients in the mean over logical cells."""
    if not samples or not len(samples) == len(advantages) == len(targets):
        raise ValueError("Incomplete crossed planner batch")
    if not 0 <= duration_weight <= 1:
        raise ValueError("Invalid duration weight")
    terms = {}
    for sample, credit, target in zip(samples, advantages, targets):
        if (not isinstance(sample, CrossedPlannerSample) or sample.policy_version != version
                or not 0 <= target < len(sample.plan.segments)):
            raise ValueError("Invalid or stale crossed planner sample")
        if not isinstance(credit, dict) or credit.keys() != {"caption", "duration"}:
            raise ValueError("Crossed credit must specify caption and duration separately")
        for field, weight in (("caption", 1 - duration_weight), ("duration", duration_weight)):
            trace, advantage = getattr(sample, field + "_trace"), credit[field]
            if not math.isfinite(advantage):
                raise ValueError("Nonfinite crossed advantage")
            if trace is None and advantage != 0:
                raise ValueError("An absent field contrast cannot have nonzero advantage")
            if trace is not None and advantage != 0 and weight > 0:
                # Each sampled axis is reused in both cells of its row or column.
                key = (id(trace), advantage)
                coefficient = terms[key][2] if key in terms else 0.
                terms[key] = (trace, advantage, coefficient + weight / len(samples))
    return list(terms.values())


def field_loss(new, old, reference, advantage, clip_epsilon, kl_beta):
    if new.ndim != 1 or new.shape != old.shape or new.shape != reference.shape or not new.numel():
        raise ValueError("Misaligned field probability trace")
    if not 0 < clip_epsilon < 1 or kl_beta < 0:
        raise ValueError("Invalid policy loss configuration")
    ratio = (new - old.detach()).exp()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage)
    delta = reference.detach() - new
    loss = (policy + kl_beta * (delta.exp() - delta - 1)).mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite crossed planner loss")
    return loss


@dataclass
class PlannerSample:
    plan: FinitePlan
    context: torch.Tensor
    response: torch.Tensor
    old_log_probs: torch.Tensor
    reference_log_probs: torch.Tensor
    constraints: dict
    decisions: dict[int, DecisionTokens]
    policy_version: int
    forced_termination: bool = False
    resource_limit_reached: bool = False
    whole_positions: tuple[int, ...] | None = None
    whole_prompt_sha256: str | None = None


def whole_mask(sample, device):
    positions = sample.whole_positions
    width = sample.response.shape[1]
    if (not positions or tuple(sorted(set(positions))) != positions or
        positions[0] < 0 or positions[-1] >= width or
        any(len(sample.constraints[i]) <= 1 for i in positions)):
        raise ValueError("Invalid whole-plan behavior decision trace")
    mask = torch.zeros(width, dtype=torch.bool, device=device)
    mask[list(positions)] = True
    return mask


class PlannerPolicy:
    def __init__(self, model_path, adapter_path, *, device="cuda:0", learning_rate=5e-7,
                 limits=PlanLimits(), base_inference=False):
        from peft import PeftModel
        self.device, self.limits, self.version = torch.device(device), limits, 0
        self.tokenizer, base = load_planner(model_path, device=self.device)
        self.base_inference = base_inference
        if base_inference:
            if adapter_path is not None:
                raise ValueError("Base planner inference must not specify an adapter")
            self.model = base.eval().requires_grad_(False)
            self.grammar = None
            return
        self.model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
        self.model.load_adapter(adapter_path, adapter_name="reference", is_trainable=False)
        for adapter in self.model.peft_config.values():
            adapter.base_model_name_or_path = str(Path(model_path).resolve())
        self.model.eval()
        self.activate("default")
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=learning_rate, weight_decay=0)
        self.grammar = None

    def activate(self, adapter):
        self.model.set_adapter(adapter)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(adapter == "default" and "lora_" in name and ".default." in name)

    @torch.no_grad()
    def sample_crossed(self, original, anchor, target, seed, *, max_new_tokens=1536):
        """Draw each axis twice with replacement; execute their Cartesian product in row-major order."""
        if getattr(self, "base_inference", False) or not 0 <= target < len(anchor.segments):
            raise ValueError("A crossed update requires an adapter and a valid anchor target")
        anchor.validate(self.limits)
        self.activate("default")
        text = generation_prefix(self.tokenizer, {"role":"user","content":instruction(original,self.limits)})
        context = self.tokenizer(text,return_tensors="pt").input_ids.to(self.device)
        rows = anchor.to_output()
        def compact(obj):
            return json.dumps(obj,separators=(",",":"),ensure_ascii=False)
        before = "[" + "".join(compact(row)+"," for row in rows[:target])
        caption_prefix = before + '{"step_index":' + str(target) + ',"prompt":"'
        duration_prefix = before + compact(rows[target]).rsplit('"num_chunks":',1)[0] + '"num_chunks":'
        def ids(prefix):
            return self.tokenizer(prefix,return_tensors="pt",add_special_tokens=False).input_ids.to(self.device)
        captions = [self._cross_caption(context,ids(caption_prefix),seed+i,max_new_tokens) for i in range(2)]
        neighbor = target + 1
        count = anchor.segments[target].num_chunks
        durations = [(count,None)] * 2
        if neighbor < len(rows):
            budget = count + anchor.segments[neighbor].num_chunks
            options = [n for n in self.limits.allowed_chunks if budget-n in self.limits.allowed_chunks]
            if len(options)>1:
                durations = self._cross_durations(torch.cat((context,ids(duration_prefix)),dim=1),options,seed+2)
        cells = []
        for caption, ct in captions:
            for duration, dt in durations:
                segments = list(anchor.segments)
                segments[target] = Segment(caption,duration)
                if neighbor < len(rows):
                    segments[neighbor] = Segment(segments[neighbor].caption,budget-duration)
                cells.append(CrossedPlannerSample(FinitePlan(tuple(segments)).validate(self.limits),
                    ct if captions[0][0]!=captions[1][0] else None,
                    dt if durations[0][0]!=durations[1][0] else None,self.version))
        return cells

    def _cross_durations(self, context, options, seed):
        support = [self.tokenizer.encode(str(n),add_special_tokens=False) for n in options]
        if any(len(t)!=1 for t in support):
            raise ValueError("A duration action must be a single tokenizer ID")
        tokens = [t[0] for t in support]
        def distribution():
            logits = self.model(input_ids=context,attention_mask=torch.ones_like(context),use_cache=False).logits
            return logits[0,-1,tokens].float().log_softmax(-1)
        old = distribution()
        self.activate("reference")
        try:
            ref = distribution()
        finally:
            self.activate("default")
        rng = torch.Generator(device=self.device).manual_seed(seed)
        chosen = torch.multinomial(old.exp(),2,replacement=True,generator=rng).tolist()
        return [(options[i],FieldTrace(context.cpu(),torch.tensor([[tokens[i]]]),{0:tokens},
                                       old[i:i+1].cpu(),ref[i:i+1].cpu())) for i in chosen]

    def _cross_caption(self, context, prefix, seed, limit):
        from transformers import GenerationConfig
        prompt = torch.cat((context,prefix),dim=1)
        if self.grammar is None:
            self.grammar = FinitePlanGrammar(self.tokenizer,context.shape[1],self.limits)
        grammar = self.grammar
        grammar.context_length, grammar.close_at = context.shape[1], None
        grammar.constraints, grammar.fields = {}, {}
        if grammar.state(prefix[0])["mode"]!="caption":
            raise ValueError("Caption prefix did not end at its field boundary")
        old = []
        pending = None
        def mask(input_ids,scores):
            nonlocal pending
            if pending is not None:
                old.append(pending[input_ids[0,-1]].clone())
            scores = grammar(input_ids,scores)
            pending = scores[0].float().log_softmax(-1)
            return scores
        def finished(input_ids,scores,**kwargs):
            state = grammar.state(input_ids[0,context.shape[1]:])
            return state["mode"] not in ("caption","escape","unicode")
        # Explicit neutral generation config: no model-specific top-k or repetition penalty.
        config = GenerationConfig(max_new_tokens=limit, do_sample=True, top_p=1.,top_k=0,temperature=1.,
                                  pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id)
        device = torch.device(self.device)
        with torch.random.fork_rng(devices=[device.index] if device.type=="cuda" else []):
            torch.manual_seed(seed)
            output = self.model.generate(prompt,attention_mask=torch.ones_like(prompt),generation_config=config,
                                         logits_processor=[mask],stopping_criteria=[finished])
        if not finished(output,None) or pending is None:
            raise ValueError("Crossed caption did not finish naturally within its token budget")
        old.append(pending[output[0,-1]])
        response = output[:,prompt.shape[1]:]
        decoded = self.tokenizer.decode(response[0],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        caption, _ = json.JSONDecoder().raw_decode('"'+decoded)
        support = {i:grammar.constraints[prefix.shape[1]+i] for i in range(response.shape[1])}
        self.activate("reference")
        try:
            ref = token_log_probs(self.model,prompt,response,support)
        finally:
            self.activate("default")
        return caption, FieldTrace(prompt.cpu(),response.cpu(),support,torch.stack(old).cpu(),ref.cpu())

    @torch.no_grad()
    def sample(self, original, seed, *, max_new_tokens=1536, text_prefix_ids=None, anchor_only=False,
               truncate_at_limit=False, whole_plan=False):
        if whole_plan and (text_prefix_ids is not None or anchor_only):
            raise ValueError("Whole-plan samples require independent unprefixed draws")
        if getattr(self, "base_inference", False):
            if not anchor_only:
                raise ValueError("Base planner is inference only")
        else:
            self.activate("default")
        text = generation_prefix(self.tokenizer, {"role": "user", "content": instruction(original, self.limits)})
        context = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        prefix = torch.empty((1, 0), dtype=torch.long, device=self.device) if text_prefix_ids is None else text_prefix_ids.to(self.device)
        prompt = torch.cat((context, prefix), dim=1)
        if self.grammar is None:
            self.grammar = FinitePlanGrammar(self.tokenizer, context.shape[1], self.limits)
        grammar = self.grammar
        grammar.close_at = prompt.shape[1] + max_new_tokens - 64 if truncate_at_limit else None
        grammar.context_length = context.shape[1]
        grammar.constraints, grammar.fields = {}, {}
        # Replay the exact prefix to record the original legal support, not a new encoding.
        for i in range(prefix.shape[1]):
            st = grammar.state(prefix[0, :i])
            allowed = grammar.allowed(st)
            if int(prefix[0, i]) not in allowed:
                raise ValueError("Provided text prefix is outside the sampling grammar")
            grammar.constraints[i], grammar.fields[i] = allowed, (len(st["lengths"]), st["mode"])
        with torch.random.fork_rng(devices=[self.device.index] if self.device.type == "cuda" else []):
            torch.manual_seed(seed)
            generated = self.model.generate(prompt, attention_mask=torch.ones_like(prompt),
                max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0, top_p=1.0,
                top_k=0, logits_processor=[grammar], pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id)
        response = generated[:, context.shape[1]:]
        if truncate_at_limit and generated.shape[1] > grammar.close_at:
            print(f'planner budget closure seed={seed} generated={generated.shape[1]-prompt.shape[1]} limit={max_new_tokens}', flush=True)
        finished = int(response[0, -1]) == self.tokenizer.eos_token_id
        text = self.tokenizer.decode(response[0], skip_special_tokens=True)
        plan = FinitePlan.parse(text, finish_reason="eos" if finished else "length", limits=self.limits)
        if anchor_only:
            return plan, response.cpu(), grammar.decision_spans(response[0].tolist(), len(plan.segments))
        old = token_log_probs(self.model, context, response, grammar.constraints).cpu()
        self.activate("reference")
        try:
            reference = token_log_probs(self.model, context, response, grammar.constraints).cpu()
        finally:
            self.activate("default")
        whole_positions = [] if whole_plan else None
        decisions = grammar.decision_spans(response[0].tolist(), len(plan.segments), whole_positions)
        cap_reached = len(plan.segments) == self.limits.max_segments or (
            plan.total_chunks + min(self.limits.allowed_chunks) > self.limits.max_total_chunks)
        return PlannerSample(plan, context.cpu(), response.cpu(), old, reference,
                             grammar.constraints.copy(), decisions, self.version,
                             forced_termination=False, resource_limit_reached=cap_reached,
                             whole_positions=tuple(whole_positions) if whole_plan else None,
                             whole_prompt_sha256=digest(original) if whole_plan else None)

    def update(self, samples, advantages, target_segments, *, clip_epsilon=0.2, kl_beta=0.01,
               duration_weight=0.5, max_grad_norm=1.0):
        from scrl.rl.objectives import planner_loss
        if not samples or len(samples) != len(advantages) or len(samples) != len(target_segments):
            raise ValueError("Incomplete planner update")
        crossed = any(isinstance(s, CrossedPlannerSample) for s in samples)
        if any(t is None for t in target_segments) and (crossed or not all(t is None for t in target_segments)):
            raise ValueError("Cannot mix whole-plan and focal policy traces")
        if crossed:
            terms = crossed_terms(samples, advantages, target_segments, version=self.version,
                                  duration_weight=duration_weight)
            if not terms:
                return {"updated": False, "losses": [], "gradient_norm": 0.0}
        elif any(s.policy_version != self.version or s.forced_termination for s in samples):
            raise ValueError("Stale or resource-truncated plan in update")
        self.activate("default")
        self.optimizer.zero_grad(set_to_none=True)
        losses = []
        try:
            observations = terms if crossed else zip(samples, advantages, target_segments)
            for sample, advantage, target_or_weight in observations:
                current = token_log_probs(self.model, sample.context.to(self.device),
                                          sample.response.to(self.device), sample.constraints)
                old, ref = sample.old_log_probs.to(self.device), sample.reference_log_probs.to(self.device)
                if crossed:
                    loss = field_loss(current, old, ref, advantage, clip_epsilon, kl_beta)
                    weight = target_or_weight
                elif target_or_weight is None:
                    mask = whole_mask(sample, self.device)
                    loss = field_loss(current[mask], old[mask], ref[mask], advantage, clip_epsilon, kl_beta)
                    weight = 1 / len(samples)
                else:
                    loss = planner_loss(current, old, ref, advantage, sample.decisions[target_or_weight],
                        clip_epsilon=clip_epsilon, kl_beta=kl_beta, duration_weight=duration_weight)
                    weight = 1 / len(samples)
                (loss * weight).backward()
                losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(self.parameters, max_grad_norm, error_if_nonfinite=True)
            if norm == 0:
                return {"updated": False, "losses": losses, "gradient_norm": 0.0}
            self.optimizer.step()
            self.version += 1
            return {"updated": True, "losses": losses, "gradient_norm": float(norm)}
        finally:
            self.optimizer.zero_grad(set_to_none=True)

    def save(self, directory):
        directory = Path(directory)
        self.activate("default")
        self.model.save_pretrained(directory / "adapter", selected_adapters=["default"])
        self.tokenizer.save_pretrained(directory / "adapter")
        torch.save(self.optimizer.state_dict(), directory / "optimizer.pt")
        (directory / "state.json").write_text(json.dumps({"planner_version": self.version}) + "\n")
