"""Bounded decoding of model decisions and observations, without default rewards."""

from itertools import product
import json


def rating_choices():
    # Every combination is available; the video-conditioned model chooses the ratings.
    return [json.dumps(dict(alignment=a, dynamics=d, physics=p, entry=None, exit=None,
                            evidence=[]), separators=(',', ':'))
            for a, d, p in product(range(5), repeat=3)]


def motion_choices():
    return [json.dumps(dict(subject_motion=s, camera_motion=c, explicit_stillness=t),
                       separators=(',', ':')) for s, c, t in product((False, True), repeat=3)]


def motion_mode_from_flags(value):
    names = {'subject_motion', 'camera_motion', 'explicit_stillness'}
    if not isinstance(value, dict) or set(value) != names or any(type(x) is not bool for x in value.values()):
        raise ValueError('Motion decisions must contain exactly three boolean flags')
    if value['subject_motion']:
        return 'mixed' if value['camera_motion'] else 'subject'
    if value['camera_motion']:
        return 'camera'
    return 'static' if value['explicit_stillness'] else 'unspecified'


def choice_constraint(tokenizer, choices, prompt_length):
    prefixes = {}
    sequences = [tokenizer.encode(text, add_special_tokens=False) for text in choices]
    for sequence in sequences:
        sequence = sequence + [tokenizer.eos_token_id]
        for index, token in enumerate(sequence):
            prefixes.setdefault(tuple(sequence[:index]), set()).add(token)

    def allowed(batch, ids):
        prefix = tuple(ids[prompt_length:].tolist())
        eos = tokenizer.eos_token_id
        if eos in prefix:
            # HF still queries the trie for finished rows while their peers generate.
            end = prefix.index(eos)
            if eos in prefixes.get(prefix[:end], ()) and all(
                    t in (eos, tokenizer.pad_token_id) for t in prefix[end+1:]):
                return [eos]
        if prefix not in prefixes:
            raise RuntimeError('Constrained rating decoder left its token trie')
        return sorted(prefixes[prefix])

    return allowed, max(map(len, sequences)) + 1


def batch_choice_constraint(tokenizer, choices_by_row, prompt_length):
    constraints = [choice_constraint(tokenizer, choices, prompt_length) for choices in choices_by_row]
    def allowed(row, ids):
        return constraints[row][0](row, ids)
    return allowed, max(limit for _, limit in constraints)


def event_batch_size(config):
    size = config.get("event_choice_batch_size", 1)
    if type(size) is not int or size not in (1, 2):
        raise ValueError("event_choice_batch_size must be 1 or 2")
    return size


class ObservationTokens:
    """A nonempty, length-bounded English observation; words are chosen by the model.

    Structural JSON and timestamps are supplied by the caller. Printable ASCII
    token pieces avoid partial UTF-8 characters and prevent empty/special-token
    answers. This constrains syntax, not which objects the model can describe.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        special = set(tokenizer.all_special_ids)
        ids = sorted(set(tokenizer.get_vocab().values()) - special)
        pieces = tokenizer.batch_decode([[i] for i in ids], skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
        self.tokens = [(i, text, sum(c.isalpha() for c in text))
                       for i, text in zip(ids, pieces)
                       if text and all(' ' <= c <= '~' for c in text)]
        if not any(text == 'a' for _, text, _ in self.tokens):
            raise ValueError('Observation tokenizer lacks a single-letter token')

    def constraint(self, prompt_length, max_chars=240):
        cache = {}

        def allowed(batch, ids):
            text = self.tokenizer.decode(ids[prompt_length:], skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
            remaining = max_chars - len(text)
            needed = max(0, 8 - sum(c.isalpha() for c in text))
            key = (remaining, needed)
            if key not in cache:
                result = [i for i, piece, letters in self.tokens
                          if len(piece) + max(0, needed - letters) <= remaining
                          and (not needed or letters)]
                if not needed:
                    result.append(self.tokenizer.eos_token_id)
                if not result:
                    raise RuntimeError('Observation decoder exhausted its character budget')
                cache[key] = result
            return cache[key]

        return allowed
