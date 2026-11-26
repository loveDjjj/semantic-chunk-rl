import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scrl.models.loading import generation_prefix, load_planner
from scrl.plan import instruction


class PlannerLoadingTests(unittest.TestCase):
    def test_legacy_and_4b_load_options(self):
        for sft, arch in ((False, "qwen3"), (False, "qwen3_5"), (True, "qwen3_5")):
            with self.subTest(sft=sft, arch=arch):
                hf = SimpleNamespace(AutoTokenizer=Mock(), AutoConfig=Mock(),
                                     AutoModelForCausalLM=Mock(), Qwen3_5ForConditionalGeneration=Mock())
                hf.AutoConfig.from_pretrained.return_value.model_type = arch
                dtype = object()
                with patch.dict(sys.modules, torch=SimpleNamespace(bfloat16=dtype), transformers=hf):
                    tok, model = load_planner("fixture", device="cpu", for_sft=sft)
                hf.AutoTokenizer.from_pretrained.assert_called_once_with(
                    "fixture", local_files_only=True, **({"padding_side": "left"} if sft else {})
                )
                self.assertIs(tok, hf.AutoTokenizer.from_pretrained.return_value)
                loader = hf.AutoModelForCausalLM if arch == "qwen3" else hf.Qwen3_5ForConditionalGeneration
                options = {"dtype": dtype} if sft else {"torch_dtype": dtype, "attn_implementation": "sdpa"}
                loader.from_pretrained.assert_called_once_with("fixture", local_files_only=True, **options)
                loader.from_pretrained.return_value.to.assert_called_once_with("cpu")
                self.assertIs(model, loader.from_pretrained.return_value.to.return_value)

    def test_template_passes_only_the_user_turn_and_disables_thinking(self):
        tokenizer = Mock()
        message = {"role": "user", "content": instruction("Water fills the glass.")}
        self.assertIs(generation_prefix(tokenizer, message), tokenizer.apply_chat_template.return_value)
        tokenizer.apply_chat_template.assert_called_once_with(
            [message], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
