"""Inkling 2 behind the server: GPT4O pre-tokenizer, typed-block OutputParser, stops, images, and template rendering.

    python -m unittest serve.test_inkling2
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate, OutputParser, openai_to_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402


class PreTokenizer(unittest.TestCase):
    def pieces(self, text):
        return ST.Tokenizer(["a"], [], pre="inkling2")._pieces(text)

    def test_gpt4o_pieces_numbers_contractions_and_case(self):
        self.assertEqual(self.pieces("1234567"), ["123", "456", "7"])
        self.assertEqual(self.pieces("don't YOU'LL I'd"), ["don't", " YOU'LL", " I'd"])
        self.assertEqual(self.pieces("MixedCASE_and\n\n  x"), ["Mixed", "CASE", "_and", "\n\n", " ", " x"])
        self.assertIsInstance(self.pieces("hello world")[0], str)


class OutputParsing(unittest.TestCase):
    TEXT = (
        "<|message_model|><|content_thinking|>Let me check <|`here`|> first.<|end_message|>"
        "<|message_model|><|content_text|>Sure, calling tools.<|end_message|>"
        "<|message_model|>weather<|content_invoke_tool_json|>{\"name\":\"weather\",\"args\":{\"city\":\"Tokyo\"}}<|end_message|>"
        "<|message_model|>clock<|content_invoke_tool_json|>{\"args\":{\"tz\":\"UTC\"}}<|end_message|>"
        "<|content_model_end_sampling|>"
    )

    def parse(self, text, step, **kw):
        p = OutputParser(thinking=True, call_format="inkling2", **kw)
        return [e for i in range(0, len(text), step) for e in p.feed(text[i:i + step])] + p.finish()

    def test_thinking_text_and_multiple_tool_calls_across_chunk_sizes(self):
        for step in (1, 3, 7, len(self.TEXT)):
            with self.subTest(step=step):
                evs = self.parse(self.TEXT, step, stream_tools=True)
                r = "".join(e.text for e in evs if e.kind == "reasoning")
                c = "".join(e.text for e in evs if e.kind == "content")
                calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
                self.assertEqual(r, "Let me check <|`here`|> first.")
                self.assertEqual(c, "Sure, calling tools.")
                self.assertEqual(calls, [("weather", {"city": "Tokyo"}), ("clock", {"tz": "UTC"})])

    def test_finish_flushes_incomplete_blocks_without_crashing(self):
        evs = self.parse("<|message_model|><|content_thinking|>half thought", 5)
        self.assertTrue(all(e.kind == "reasoning" for e in evs))
        self.assertEqual("".join(e.text for e in evs), "half thought")
        evs = self.parse("<|message_model|><|content_text|>half answer<|end_mes", 4)
        self.assertTrue(all(e.kind == "content" for e in evs))
        self.assertEqual("".join(e.text for e in evs), "half answer<|end_mes")
        evs = self.parse("<|message_model|>ping<|content_invoke_tool_json|>{\"name\":\"ping\",\"args\":{\"x\":1}}", 6)
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].kind, "tool_call")
        self.assertEqual((evs[0].call.name, evs[0].call.arguments), ("ping", {"x": 1}))


class Inkling2Tokens(ByteTokenizer):
    SPECIALS = [
        "<|message_user|>",
        "<|message_model|>",
        "<|message_system|>",
        "<|message_tool|>",
        "<|content_thinking|>",
        "<|content_text|>",
        "<|content_invoke_tool_json|>",
        "<|content_image|>",
        "<|unused_200054|>",
        "<|end_message|>",
        "<|content_model_end_sampling|>",
    ]
    pre = "inkling2"
    special_ids = {"tokenizer.ggml.eos_token_id": 266, "tokenizer.ggml.eot_token_id": 266}


MINI_TEMPLATE = Path(r"D:\Users\Kurryn_PC\Documents\Strata\Models\packs\inkling2-ud-q4_k_xl\tokenizer\chat_template.jinja")


class ServiceForInkling2(unittest.TestCase):
    def setUp(self):
        tok = Inkling2Tokens()
        self._tmp = tempfile.TemporaryDirectory()
        tpl_path = Path(self._tmp.name) / "chat_template.jinja"
        if MINI_TEMPLATE.exists():
            tpl_path.write_text(MINI_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            tpl_path.write_text(
                "{%- if reasoning_effort is defined and reasoning_effort == 'none' -%}0{%- else -%}0.9{%- endif -%}"
                "{%- for m in messages -%}<|message_user|><|content_text|>{{ m.content }}<|end_message|>{%- endfor -%}",
                encoding="utf-8",
            )
        self.svc = Service(MockEngine(tok, "hi"), tok, ChatTemplate(tpl_path))

    def tearDown(self):
        self._tmp.cleanup()

    def test_stops_turns_call_format_and_image_marks(self):
        self.assertEqual(self.svc.call_format, "inkling2")
        self.assertIn(266, self.svc.stop_ids)
        self.assertEqual(self.svc.turn_ids, frozenset({256, 257, 258, 259}))
        self.assertEqual(self.svc.image_marks, ("<|unused_200054|>", "<|content_image|>", None))

    @unittest.skipUnless(MINI_TEMPLATE.exists(), "Inkling 2 template file not downloaded")
    def test_template_efforts_tool_history_and_effort_end_bypass(self):
        self.svc.effort_end = True
        msgs, tools, kw = openai_to_messages({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "none"})
        rendered_none = self.svc.render_prompt(msgs, tools, kw)
        self.assertIn("Thinking effort level: 0<|end_message|>", rendered_none)

        msgs, tools, kw = openai_to_messages({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low"})
        rendered_low = self.svc.render_prompt(msgs, tools, kw)
        self.assertIn("Thinking effort level: 0.2<|end_message|>", rendered_low)
        self.assertNotIn("Thinking effort:", rendered_low)

        raw_msgs = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "20C"},
            {"role": "user", "content": "thanks"},
        ]
        rendered_tc = self.svc.render_prompt(raw_msgs, None, {})
        self.assertIn("<|message_model|>get_weather<|content_invoke_tool_json|>", rendered_tc)
        self.assertIn('"args":{"city":"Paris"}', rendered_tc)
        self.assertIn("<|message_tool|>get_weather<|content_text|>20C<|end_message|>", rendered_tc)


if __name__ == "__main__":
    unittest.main()
