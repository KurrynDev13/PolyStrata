"""Tests for setup.py's Inkling-2-Mini: its four pinned files, fork prebuilt URL, KV cache formats, and install on
strata-poly with no image encoder. Mocked like tools/test_setup_glm.py, whose harness this uses.

    python -m unittest tools.test_setup_inkling2
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools import test_setup_glm as glm_tests  # noqa: E402

FAM = setup.FAMILIES["inkling2"]
SIZES = [
    ("UD-Q3_K_XL", 13631778016),
    ("UD-IQ3_XXS", 11663309504),
    ("UD-Q4_K_XL", 17629759456),
    ("UD-Q8_K_XL", 22867451616),
]


class Pins(unittest.TestCase):
    def test_the_files(self):
        self.assertEqual(list(setup.family_models("inkling2")), [m for m, _ in SIZES])
        self.assertRegex(FAM["hf"], r"^https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/[0-9a-f]{40}/$")
        self.assertEqual(list(FAM["sha256"]), [f"Inkling-2-Mini-{m}.gguf" for m, _ in SIZES])
        for size, n in SIZES:
            file = f"Inkling-2-Mini-{size}.gguf"
            d = FAM["models"][size]
            self.assertEqual((setup.model_file(FAM, size, 1), setup.model_shards(FAM, size)), (file, 1))
            self.assertEqual(FAM["sha256"][file][0], n)
            self.assertRegex(FAM["sha256"][file][1], r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(n / 1e9, d["download_gb"], delta=0.05)
            self.assertLess(d["arena_gb"], d["download_gb"])
            self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)), size)
        self.assertEqual(setup.POLY_GGUF_ARCHS["inkling2"], ("inkling2_loop", "inkling2"))
        self.assertIn("KurrynDev13/PolyStrata-Inkling2", setup.POLY_PREBUILT_URL)

    def test_the_about_line_says_the_first_size(self):
        n = setup.poly_ram_need(FAM["models"]["UD-Q3_K_XL"], 8.0)
        self.assertIn(f"~{n:.0f} GB of RAM beside an 8 GB card", FAM["about"])


class Main(unittest.TestCase):
    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def test_install_each_size(self):
        for size, _n in SIZES:
            with self.subTest(size):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, _ = self.main(family="inkling2", model=size, ram=64.0, vram=8188 / 1024)
                self.assertEqual(code, 0, out)
                cfg = json.loads((self.t / f"strata-inkling2-{size.lower()}.json").read_text())
                file = f"Inkling-2-Mini-{size}.gguf"
                self.assertEqual(self.downloads, [FAM["hf"] + file])
                self.assertEqual(self.verified, [(file, *FAM["sha256"][file])])
                tag = "inkling2-" + size
                pack = self.t / "data" / "packs" / tag.lower()
                self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
                args = cfg["args"]
                self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / tag / file)
                self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
                for flag in ("--draft-model", "--tail-skip"):
                    self.assertNotIn(flag, args)
                self.assertEqual(cfg["model_name"], f"inkling-2-mini-{size.lower()}")
                self.assertEqual(cfg["display_name"], f"Inkling-2-Mini {size}")
                self.assertNotIn("vision", cfg)

    def test_kv_formats_go_into_config_and_persist(self):
        for kv in ("k8v4", "q8_0", "q4_0"):
            with self.subTest(kv):
                self.tearDown()
                self.setUp()
                code, out, cfg = self.main(["--kv", kv], family="inkling2", model="UD-Q3_K_XL", ram=32.0, vram=12.0)
                self.assertEqual(code, 0, out)
                self.assertEqual(cfg["args"][cfg["args"].index("--kv") + 1], kv)
                self.assertNotIn("not used with", out)
                code, out, cfg = self.main(family="inkling2", model="UD-Q3_K_XL", ram=32.0, vram=12.0)
                self.assertEqual(cfg["args"][cfg["args"].index("--kv") + 1], kv)
                code, out, cfg = self.main(["--kv", "fp16"], family="inkling2", model="UD-Q3_K_XL", ram=32.0, vram=12.0)
                self.assertEqual(code, 0, out)
                self.assertNotIn("--kv", cfg["args"])

    def test_unsupported_kv_and_vision_warn(self):
        code, out, cfg = self.main(["--kv", "int8"], family="inkling2", model="UD-Q3_K_XL", ram=32.0, vram=12.0)
        self.assertEqual(code, 0, out)
        self.assertIn("--kv", next(ln for ln in out.splitlines() if "not used with Inkling-2-Mini" in ln))
        self.assertNotIn("--kv", cfg["args"])

        code, out, cfg = self.main(["--vision", "yes"], family="inkling2", model="UD-Q3_K_XL", ram=32.0, vram=12.0)
        self.assertEqual(code, 0, out)
        self.assertIn("not used with Inkling-2-Mini", out)
        self.assertNotIn("vision", cfg)


if __name__ == "__main__":
    unittest.main()
