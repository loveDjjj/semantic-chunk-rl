"""Serializable contracts shared by model backends and the coordinator."""

from dataclasses import asdict, dataclass
import hashlib
import json


def digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class PlanLimits:
    allowed_chunks: tuple[int, ...] = (1, 2, 4, 6)
    max_segments: int = 5
    max_total_chunks: int = 30

    def __post_init__(self):
        if not isinstance(self.allowed_chunks, (list, tuple)) or not self.allowed_chunks or any(
                type(n) is not int or n < 1 for n in self.allowed_chunks):
            raise ValueError("Chunk support must contain positive integers")
        object.__setattr__(self, "allowed_chunks", tuple(self.allowed_chunks))
        if len(set(self.allowed_chunks)) != len(self.allowed_chunks):
            raise ValueError("Duplicate chunk choices")
        if type(self.max_segments) is not int or self.max_segments < 1:
            raise ValueError("Invalid segment limit")
        if type(self.max_total_chunks) is not int or self.max_total_chunks < 1:
            raise ValueError("Invalid total chunk limit")
        if min(self.allowed_chunks) > self.max_total_chunks:
            raise ValueError("No allowed chunk count fits the total chunk limit")


@dataclass(frozen=True)
class Segment:
    caption: str
    num_chunks: int

    def __post_init__(self):
        if not isinstance(self.caption, str) or not self.caption.strip():
            raise ValueError("Empty segment caption")
        if type(self.num_chunks) is not int or self.num_chunks < 1:
            raise ValueError("Chunk count must be a positive integer")

    @property
    def latent_frames(self):
        return 3 * self.num_chunks


@dataclass(frozen=True)
class FinitePlan:
    segments: tuple[Segment, ...]

    def __post_init__(self):
        if not isinstance(self.segments, tuple) or not self.segments:
            raise ValueError("A plan needs at least one segment")
        if not all(isinstance(x, Segment) for x in self.segments):
            raise TypeError("Expected Segment objects")

    def validate(self, limits: PlanLimits):
        if len(self.segments) > limits.max_segments:
            raise ValueError("Plan exceeds the segment resource limit")
        if self.total_chunks > limits.max_total_chunks:
            raise ValueError("Plan exceeds the chunk resource limit")
        if any(s.num_chunks not in limits.allowed_chunks for s in self.segments):
            raise ValueError("Unsupported joint group length")
        return self

    @property
    def total_chunks(self):
        return sum(s.num_chunks for s in self.segments)

    @property
    def fingerprint(self):
        return digest(self.to_dict())

    def to_dict(self):
        """Internal serialization; model output uses the original-style JSON array."""
        return {"segments": [asdict(s) for s in self.segments]}

    def to_output(self):
        return [{"step_index": index, "prompt": segment.caption, "num_chunks": segment.num_chunks}
                for index, segment in enumerate(self.segments)]

    @classmethod
    def from_dict(cls, value, limits=PlanLimits()):
        if not isinstance(value, dict) or set(value) != {"segments"}:
            raise ValueError("Expected exactly segments")
        if not isinstance(value["segments"], list):
            raise ValueError("segments must be a list")
        segments = []
        for row in value["segments"]:
            if not isinstance(row, dict) or set(row) != {"caption", "num_chunks"}:
                raise ValueError("Each segment requires exactly caption and num_chunks")
            segments.append(Segment(**row))
        return cls(tuple(segments)).validate(limits)

    @classmethod
    def parse(cls, text, *, finish_reason="stop", limits=PlanLimits()):
        if finish_reason not in {"stop", "eos"}:
            raise ValueError("Truncated or interrupted output is not a complete plan")
        def no_duplicates(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON field")
                result[key] = value
            return result
        rows = json.loads(text, object_pairs_hook=no_duplicates)
        if not isinstance(rows, list) or not rows:
            raise ValueError("Planner output must be a nonempty JSON array")
        segments = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {"step_index", "prompt", "num_chunks"}:
                raise ValueError("Each output row requires step_index, prompt and num_chunks")
            if type(row["step_index"]) is not int or row["step_index"] != index:
                raise ValueError("step_index must be consecutive and zero-based")
            segments.append(Segment(row["prompt"], row["num_chunks"]))
        return cls(tuple(segments)).validate(limits)

    def spans(self):
        cursor = 0
        rows = []
        for index, segment in enumerate(self.segments):
            end = cursor + segment.latent_frames
            rows.append(SegmentSpan(index, cursor, end, max(0, 4 * cursor - 3), 4 * end - 3))
            cursor = end
        return tuple(rows)


@dataclass(frozen=True)
class SegmentSpan:
    index: int
    latent_start: int
    latent_end: int
    pixel_start: int
    pixel_end: int


@dataclass(frozen=True)
class PolicyStamp:
    planner: int = 0
    executor: int = 0
    reward: str = "quality-semantic-v1"
    sampler: str = "indexed-noise-v1"
    planner_reference: str = "initial-planner"
    executor_reference: str = "initial-executor"

    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in (self.planner, self.executor)):
            raise ValueError("Invalid policy version")
        if not all(isinstance(x, str) and x for x in
                   (self.reward, self.sampler, self.planner_reference, self.executor_reference)):
            raise ValueError("Missing policy identity")

    @property
    def fingerprint(self):
        return digest(asdict(self))


def instruction(original: str, limits=PlanLimits()) -> str:
    if not original.strip():
        raise ValueError("Empty instruction")
    return (
        'You are an expert in temporal reasoning and cinematic prompt planning for video generation.\n'
        'Given the original video instruction, produce its complete temporal plan in one response. '
        'Choose the number of segments yourself according to the content: there is no fixed number '
        'of steps and no requirement to split time evenly. Include EVERY requested event and its '
        'visible result, in the requested order. Do not stop the plan after only the first event.\n'
        'Each segment describes one coherent action, phase of an action, or maintained state. '
        'Write a visually descriptive present-tense caption that a video model can execute. '
        'Preserve the original subjects, identities, colors, counts, setting and object relationships. '
        'Describe only the active phase, without performing a future action too early. Do not '
        'invent extra actions or objects. A static scene may use one segment; a multi-event '
        'instruction usually needs multiple segments. Keep the onset, continuous motion and visible '
        'result of the SAME action together when they form one coherent motion. Start a new '
        'segment when the active action or requested state changes, not for each successive '
        'snapshot. Do not append repeated aftermath or idle-hold segments after the task is done '
        'unless the original instruction explicitly asks for them.\n'
        'Also choose num_chunks for each segment. All chunks assigned to a segment are denoised '
        'JOINTLY as one group. The number controls both its video duration and joint generation '
        f'range. Allowed num_chunks values are {list(limits.allowed_chunks)}. '
        'One native chunk is roughly 0.75 seconds of video: 1 chunk is brief, 2 about 1.5 seconds, '
        '4 about 3 seconds and 6 about 4.5 seconds. Choose enough time for the action in that '
        'segment. A brief change and a sustained motion need not have the same length. '
        'As an illustration only, "open a box, slowly draw a ribbon out, then close the box" '
        'can use three segments with 1, 4 and 2 chunks. A still vase can use a single segment. '
        'These are examples of granularity, not a fixed plan or a rule to copy their objects '
        'or lengths into other instructions. There is no fixed total chunk sum and no reward '
        'for filling a budget. '
        f'Use at most {limits.max_segments} segments and at most {limits.max_total_chunks} total chunks.\n'
        'Return ONLY a JSON array. Each object has exactly three keys: "step_index" '
        '(consecutive integer starting at 0), "prompt" (the segment caption), and "num_chunks" '
        '(an allowed integer). Finish the array when all requested content has been planned. '
        'Do not add a terminal field, a special end marker, markdown, or explanatory text.\n'
        f'ORIGINAL INSTRUCTION:\n{original}'
    )


@dataclass(frozen=True)
class DecisionTokens:
    """Half-open spans in original response token IDs, never re-tokenized text."""
    caption: tuple[int, int]
    duration: tuple[int, int]

    def masks(self, length, device=None):
        import torch
        result = []
        for begin, end in (self.caption, self.duration):
            if not 0 <= begin < end <= length:
                raise ValueError("Decision token span is outside the response")
            mask = torch.zeros(length, dtype=torch.bool, device=device)
            mask[begin:end] = True
            result.append(mask)
        if (result[0] & result[1]).any():
            raise ValueError("Caption and duration token spans overlap")
        return tuple(result)
