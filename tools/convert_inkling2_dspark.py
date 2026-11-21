#!/usr/bin/env python3
"""Convert thinkingmachines/Inkling-2-Mini mtp.safetensors into a standalone DSpark GGUF.

Downloads config.json and mtp.safetensors from HuggingFace (unless --source-dir already holds
them), validates mtp_config and all 54 BF16 safetensors entries, writes a GGUF v3 file with
unmodified BF16 payloads (shape[0] = contiguous inner axis), and verifies the written GGUF with
tools/gguf_reader.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import pathlib
import struct
import sys
import tempfile
import urllib.request

_TOOLS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR))
from gguf_reader import GGUF_MAGIC, GGUFFile  # noqa: E402

HF_BASE = "https://huggingface.co/{repo}/resolve/main/{filename}"
DEFAULT_OUTPUT = (
    r"D:\Users\Kurryn_PC\Documents\Strata\Models\models\inkling2-UD-Q4_K_XL\inkling2-dspark.gguf"
)
DEFAULT_ALIGNMENT = 32
TYPE_BF16 = 30
META_IDS = {"u32": 4, "f32": 6, "string": 8, "array": 9}


def download_file(url: str, dest: pathlib.Path) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "PolyStrata-DSpark-Converter/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        if dest.exists() and total > 0 and dest.stat().st_size == total:
            print(f"Using cached {dest} ({total:,} bytes)")
            return dest
        tmp = dest.with_suffix(dest.suffix + ".part")
        done = 0
        chunk = 4 * 1024 * 1024
        with open(tmp, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total > 0:
                    pct = 100.0 * done / total
                    print(f"\rDownloading {dest.name}: {done / (1 << 20):.1f}/{total / (1 << 20):.1f} MiB ({pct:.1f}%)",
                          end="", flush=True)
        if total > 0:
            print()
            if done != total:
                raise RuntimeError(f"Short download for {url}: got {done} of {total} bytes")
        os.replace(tmp, dest)
    return dest


def pack_meta_kv(key: str, typ: str, value) -> bytes:
    kb = key.encode("utf-8")
    out = bytearray(struct.pack("<Q", len(kb)) + kb)

    def pack_scalar(t: str, v) -> bytes:
        if t == "string":
            b = v.encode("utf-8")
            return struct.pack("<Q", len(b)) + b
        if t == "u32":
            return struct.pack("<I", int(v))
        if t == "f32":
            return struct.pack("<f", float(v))
        raise ValueError(f"Unsupported scalar metadata type: {t}")

    if typ.startswith("array:"):
        elem = typ.split(":", 1)[1]
        out += struct.pack("<I", META_IDS["array"])
        out += struct.pack("<I", META_IDS[elem])
        out += struct.pack("<Q", len(value))
        for item in value:
            out += pack_scalar(elem, item)
        return bytes(out)

    out += struct.pack("<I", META_IDS[typ])
    out += pack_scalar(typ, value)
    return bytes(out)


def read_safetensors_header(path: pathlib.Path) -> tuple[int, dict]:
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        hdr_len_bytes = f.read(8)
        if len(hdr_len_bytes) != 8:
            raise ValueError(f"{path} is truncated")
        (hdr_len,) = struct.unpack("<Q", hdr_len_bytes)
        if 8 + hdr_len > file_size:
            raise ValueError(f"{path} header length {hdr_len} exceeds file size {file_size}")
        hdr = json.loads(f.read(hdr_len).decode("utf-8"))
    return 8 + hdr_len, hdr


def build_metadata(cfg: dict, markov_vocab: int) -> list[tuple[str, str, object]]:
    mtp = cfg.get("mtp_config") or cfg
    arch = "inkling2_dspark"
    return [
        ("general.architecture", "string", arch),
        ("general.name", "string", "Inkling-2-Mini-DSpark"),
        (f"{arch}.block_count", "u32", int(mtp["num_hidden_layers"])),
        (f"{arch}.embedding_length", "u32", int(mtp["hidden_size"])),
        (f"{arch}.feed_forward_length", "u32", int(mtp["intermediate_size"])),
        (f"{arch}.attention.head_count", "u32", int(mtp["num_attention_heads"])),
        (f"{arch}.attention.head_count_kv", "u32", int(mtp["num_key_value_heads"])),
        (f"{arch}.attention.key_length", "u32", int(mtp["head_dim"])),
        (f"{arch}.attention.value_length", "u32", int(mtp["head_dim"])),
        (f"{arch}.attention.layer_norm_rms_epsilon", "f32", float(mtp["rms_norm_eps"])),
        (f"{arch}.attention.sliding_window", "u32", int(mtp["sliding_window_size"])),
        (f"{arch}.rope.freq_base", "f32", float(mtp["rope_theta"])),
        (f"{arch}.context_length", "u32", int(mtp["max_position_embeddings"])),
        (f"{arch}.block_size", "u32", int(mtp["block_size"])),
        (f"{arch}.markov_rank", "u32", int(mtp["markov_rank"])),
        (f"{arch}.markov_head_type", "string", str(mtp["markov_head_type"])),
        (f"{arch}.vocab_size", "u32", int(markov_vocab)),
        (f"{arch}.target_layer_ids", "array:u32", [int(x) for x in mtp["target_layer_ids"]]),
        (f"{arch}.local_layer_ids", "array:u32", [int(x) for x in mtp["local_layer_ids"]]),
        ("general.alignment", "u32", DEFAULT_ALIGNMENT),
    ]


def convert(config_path: pathlib.Path, safetensors_path: pathlib.Path, output_path: pathlib.Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    mtp = cfg.get("mtp_config") or cfg
    if mtp.get("markov_head_type") != "rnn_hidden":
        raise ValueError(f"Expected markov_head_type='rnn_hidden', got {mtp.get('markov_head_type')!r}")

    data_base, hdr = read_safetensors_header(safetensors_path)
    hdr.pop("__metadata__", None)
    names = sorted(hdr.keys())
    if len(names) != 54:
        raise ValueError(f"Expected 54 DSpark tensors in {safetensors_path}, got {len(names)}")

    markov_w1 = hdr["markov_head.markov_w1.weight"]
    markov_vocab = int(markov_w1["shape"][0])
    metadata = build_metadata(cfg, markov_vocab)

    tensors = []
    rel_offset = 0
    for name in names:
        info = hdr[name]
        if info["dtype"] != "BF16":
            raise ValueError(f"Tensor {name} has dtype {info['dtype']}, expected BF16")
        sf_shape = [int(d) for d in info["shape"]]
        # Safetensors C row-major [out, in] already has `in` contiguous along memory,
        # matching GGUF shape[0] = in, shape[1] = out without transposing data bytes.
        gguf_shape = list(reversed(sf_shape)) if len(sf_shape) >= 2 else sf_shape
        start, end = int(info["data_offsets"][0]), int(info["data_offsets"][1])
        nbytes = end - start
        elems = 1
        for d in sf_shape:
            elems *= d
        if nbytes != elems * 2:
            raise ValueError(f"Byte length mismatch for {name}: {nbytes} vs {elems * 2}")
        tensors.append((name, gguf_shape, start, nbytes, rel_offset))
        rel_offset += nbytes
        rel_offset = (rel_offset + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT * DEFAULT_ALIGNMENT

    header_buf = bytearray()
    header_buf += struct.pack("<I", GGUF_MAGIC)
    header_buf += struct.pack("<I", 3)
    header_buf += struct.pack("<Q", len(tensors))
    header_buf += struct.pack("<Q", len(metadata))
    for k, typ, val in metadata:
        header_buf += pack_meta_kv(k, typ, val)

    for name, gguf_shape, _, _, offset in tensors:
        nb = name.encode("utf-8")
        header_buf += struct.pack("<Q", len(nb)) + nb
        header_buf += struct.pack("<I", len(gguf_shape))
        header_buf += struct.pack(f"<{len(gguf_shape)}Q", *gguf_shape)
        header_buf += struct.pack("<I", TYPE_BF16)
        header_buf += struct.pack("<Q", offset)

    pad = (-len(header_buf)) % DEFAULT_ALIGNMENT
    header_buf += b"\0" * pad

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_path.with_suffix(".gguf.tmp")
    src_hashes: dict[str, bytes] = {}
    with open(safetensors_path, "rb") as fin, open(tmp_out, "wb") as fout:
        mm = mmap.mmap(fin.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            with memoryview(mm) as mv:
                fout.write(header_buf)
                for name, _, start, nbytes, _ in tensors:
                    view = mv[data_base + start: data_base + start + nbytes]
                    src_hashes[name] = hashlib.sha256(view).digest()
                    fout.write(view)
                    del view
                    tpad = (-nbytes) % DEFAULT_ALIGNMENT
                    if tpad:
                        fout.write(b"\0" * tpad)
        finally:
            mm.close()
    os.replace(tmp_out, output_path)
    verify_gguf(output_path, hdr, src_hashes, metadata)


def verify_gguf(
    gguf_path: pathlib.Path,
    hdr: dict,
    src_hashes: dict[str, bytes],
    expected_meta: list[tuple[str, str, object]],
) -> None:
    g = GGUFFile(gguf_path)
    tmap = {t.name: t for t in g.tensors}
    if len(tmap) != len(hdr):
        raise AssertionError(f"GGUF tensor count {len(tmap)} != {len(hdr)}")
    for k, _, val in expected_meta:
        got = g.metadata.get(k)
        if isinstance(val, float):
            if abs(float(got) - val) > 1e-6 * max(1.0, abs(val)):
                raise AssertionError(f"Metadata {k}: got {got!r}, expected {val!r}")
        elif got != val:
            raise AssertionError(f"Metadata {k}: got {got!r}, expected {val!r}")

    with open(gguf_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            with memoryview(mm) as mv:
                for name, info in sorted(hdr.items()):
                    ti = tmap[name]
                    if ti.type_name != "BF16":
                        raise AssertionError(f"{name}: expected BF16, got {ti.type_name}")
                    sf_shape = [int(d) for d in info["shape"]]
                    exp_shape = list(reversed(sf_shape)) if len(sf_shape) >= 2 else sf_shape
                    if ti.shape != exp_shape:
                        raise AssertionError(f"{name}: shape {ti.shape} != expected {exp_shape}")
                    nbytes = ti.expected_bytes()
                    start = g.data_start + ti.offset
                    view = mv[start: start + nbytes]
                    got_hash = hashlib.sha256(view).digest()
                    del view
                    if got_hash != src_hashes[name]:
                        raise AssertionError(f"{name}: SHA-256 mismatch after write")
        finally:
            mm.close()
    print(f"Verified {gguf_path} ({gguf_path.stat().st_size:,} bytes, {len(g.tensors)} BF16 tensors, SHA-256 exact match)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert Inkling-2-Mini mtp.safetensors to GGUF")
    ap.add_argument("--repo", default="thinkingmachines/Inkling-2-Mini")
    ap.add_argument("--source-dir", type=pathlib.Path, default=None,
                    help="Local folder containing config.json and mtp.safetensors (skips download)")
    ap.add_argument("--cache-dir", type=pathlib.Path,
                    default=pathlib.Path(tempfile.gettempdir()) / "inkling2-dspark-hf")
    ap.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path(DEFAULT_OUTPUT))
    args = ap.parse_args()

    if args.source_dir is not None:
        cfg_path = args.source_dir / "config.json"
        mtp_path = args.source_dir / "mtp.safetensors"
    else:
        cfg_path = download_file(HF_BASE.format(repo=args.repo, filename="config.json"),
                                 args.cache_dir / "config.json")
        mtp_path = download_file(HF_BASE.format(repo=args.repo, filename="mtp.safetensors"),
                                 args.cache_dir / "mtp.safetensors")

    convert(cfg_path, mtp_path, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
