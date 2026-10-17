"""Run Qwen3-VL-4B-Instruct on video files, locally or via Alibaba Cloud Model Studio.

Local (GPU, 4-bit):
    .venv\\Scripts\\python.exe infer_video.py clip.mp4 --prompt "Describe this video."
    .venv\\Scripts\\python.exe infer_video.py a.mov b.mov --system-file prompts\\cashier_ru.txt --crop

API (needs DASHSCOPE_API_KEY):
    .venv\\Scripts\\python.exe infer_video.py clip.mp4 --prompt "Describe this video." --api
    .venv\\Scripts\\python.exe infer_video.py clip.mp4 --api --send frames

Several videos can be passed at once; the local model is loaded only once.
--crop cuts each video down to the region with motion (see crop.py) before inference.
Local mode loads the model in 4-bit (bitsandbytes NF4) so it fits a 4 GB GPU.
"""
import argparse
import base64
import io
import json
import os
import re
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
# Same weights re-split into ~400 MB shards: Windows reserves commit memory for each
# whole memory-mapped shard, so small shards keep peak memory low while loading.
MODEL_DIR = os.path.join(HERE, "models", "Qwen3-VL-4B-Instruct-small")
CROP_DIR = os.path.join(HERE, "cropped")

API_BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
API_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3-vl-4b-instruct")
# Model Studio rejects base64 payloads much above ~10 MB; larger videos go as frames.
MAX_INLINE_VIDEO_BYTES = 7 * 2**20

DEFAULT_PROMPT = "Describe this video in detail."
DEFAULT_PROMPT_WITH_SYSTEM = "Проанализируй видео и верни JSON."
# --think: small models miss violations when forced to answer with JSON directly,
# so let them write a timeline and check it before the final JSON.
THINK_PROMPT = (
    "Сначала кратко опиши хронологию видео по секундам: что делает кассир; состояние ноутбука "
    "(открыт/закрыт, экран включён); сейфа/шкафа и ключа; денег, сумок, документов, штампов; двери; "
    "когда кассир покидает рабочее место и в каком состоянии всё остаётся в этот момент. "
    "Затем пройди по категориям нарушений и для каждой подходящей проверь, подтверждает ли её хронология. "
    "В самом конце выведи итоговый JSON строго в требуемом формате."
)


# ---------------------------------------------------------------- helpers

def prepare_video(path, crop):
    """Return the path to feed the model: the original, or a cached cropped copy."""
    if not crop:
        return path
    from crop import find_action_box, write_cropped

    os.makedirs(CROP_DIR, exist_ok=True)
    out = os.path.join(CROP_DIR, os.path.splitext(os.path.basename(path))[0] + ".crop.mp4")
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path):
        print(f"[crop: using cached {out}]", file=sys.stderr)
        return out
    box, _ = find_action_box(path)
    x0, y0, x1, y1 = box
    print(f"[crop: x={x0}..{x1} y={y0}..{y1} ({x1 - x0}x{y1 - y0})]", file=sys.stderr)
    return write_cropped(path, box, out)


def extract_json(text):
    """Return the last JSON object in the answer that has a "verdict" key.

    Tolerates ``` fences and reasoning text before it (--think)."""
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r"\{", text))):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            return obj
    return None


def report(name, answer, want_json, show_reasoning=False):
    print(f"===== {name} =====")
    if want_json:
        parsed = extract_json(answer)
        if parsed is not None:
            if show_reasoning:
                print(answer[:answer.rfind('"verdict"')].rstrip(" {\n`json"), file=sys.stderr)
            print(json.dumps(parsed, ensure_ascii=False, indent=2))
            return
        print("[warning: answer is not valid JSON, raw output follows]", file=sys.stderr)
    print(answer)


# ---------------------------------------------------------------- local (GPU)

def load_model(model_dir=MODEL_DIR, precision="4bit"):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if precision == "4bit":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # GTX 16xx (Turing) has no native bf16; fp16 compute is much faster there.
            bnb_4bit_compute_dtype=torch.float16,
        )
        dtype = torch.float16
    elif precision == "8bit":  # ~half the VRAM of bf16, near-bf16 quality
        quant, dtype = BitsAndBytesConfig(load_in_8bit=True), torch.bfloat16
    else:  # full precision, for GPUs with enough VRAM (Ampere+ for bf16)
        quant, dtype = None, torch.bfloat16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        quantization_config=quant,
        dtype=dtype,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_dir)
    return model, processor


def ask_video(model, processor, video_path, prompt, system=None, fps=1.0, max_frames=16,
              max_pixels=320 * 240, max_new_tokens=512, repetition_penalty=None):
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": os.path.abspath(video_path),
                "fps": fps,
                "max_frames": max_frames,
                # Per-frame pixel budget; the main lever for VRAM use.
                "max_pixels": max_pixels,
                # qwen-vl-utils' default video min_pixels exceeds small budgets.
                "min_pixels": min(max_pixels, 128 * 32 * 32) // 4,
            },
            {"type": "text", "text": prompt},
        ],
    })

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )
    video_metadata = None
    if videos is not None:
        videos, video_metadata = map(list, zip(*videos))

    inputs = processor(
        text=[text], images=images, videos=videos, video_metadata=video_metadata,
        do_resize=False, cap_pixels_per_frame=True, return_tensors="pt", **video_kwargs,
    ).to(model.device)

    with torch.inference_mode():
        # Greedy decoding can loop on long free-text fields; a mild penalty breaks the loop.
        extra = {"repetition_penalty": repetition_penalty} if repetition_penalty else {}
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, **extra)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0], inputs["input_ids"].shape[1]


def run_local(args, jobs):
    import torch

    t0 = time.time()
    model, processor = load_model(args.model_dir, args.precision)
    print(f"[loaded in {time.time() - t0:.1f}s, "
          f"VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GB]", file=sys.stderr)

    for name, path in jobs:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        answer, n_tokens = ask_video(model, processor, path, args.prompt, args.system, args.fps,
                                     args.max_frames, args.max_pixels, args.max_new_tokens)
        print(f"[{name}: {n_tokens} input tokens, {time.time() - t0:.1f}s, "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB]", file=sys.stderr)
        report(name, answer, args.system is not None, args.think)


# ---------------------------------------------------------------- API (Model Studio)

def sample_frames_as_data_urls(video_path, fps, max_frames, max_pixels):
    """Decode the video with PyAV and return JPEG data URLs for evenly spaced frames."""
    import av

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else None
        frames = [f.to_image() for f in container.decode(stream)]
    if not frames:
        sys.exit(f"No frames decoded from {video_path}")

    if duration is None:
        duration = len(frames) / float(stream.average_rate or 24)
    count = max(2, min(max_frames, round(duration * fps), len(frames)))
    picks = [frames[round(i * (len(frames) - 1) / (count - 1))] for i in range(count)]

    urls = []
    for img in picks:
        scale = min(1.0, (max_pixels / (img.width * img.height)) ** 0.5)
        if scale < 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        urls.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    return urls


def run_api(args, jobs):
    from openai import OpenAI

    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        sys.exit("DASHSCOPE_API_KEY is not set. See README.md for setup.")
    client = OpenAI(api_key=key, base_url=API_BASE_URL)

    for name, path in jobs:
        send = args.send
        if send == "auto":
            send = "video" if os.path.getsize(path) <= MAX_INLINE_VIDEO_BYTES else "frames"

        if send == "video":
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            mime = "video/quicktime" if path.lower().endswith(".mov") else "video/mp4"
            media = {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{data}"}, "fps": args.fps}
        else:
            frames = sample_frames_as_data_urls(path, args.fps, args.max_frames, args.max_pixels)
            media = {"type": "video", "video": frames}

        messages = [{"role": "system", "content": args.system}] if args.system else []
        messages.append({"role": "user", "content": [media, {"type": "text", "text": args.prompt}]})

        t0 = time.time()
        resp = client.chat.completions.create(model=args.model, messages=messages,
                                              max_tokens=args.max_new_tokens)
        u = resp.usage
        print(f"[{name}: {args.model} via {send}, {u.prompt_tokens} in / {u.completion_tokens} out tokens, "
              f"{time.time() - t0:.1f}s]", file=sys.stderr)
        report(name, resp.choices[0].message.content, args.system is not None, args.think)


# ---------------------------------------------------------------- CLI

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--prompt", help="user message (default depends on --system-file)")
    ap.add_argument("--system-file", help="text file with the system prompt; enables JSON output parsing")
    ap.add_argument("--crop", action="store_true", help="crop each video to the region with motion first")
    ap.add_argument("--think", action="store_true",
                    help="with --system-file: let the model write a timeline before the JSON (more accurate, slower)")
    ap.add_argument("--fps", type=float, default=1.0, help="frames sampled per second of video")
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--max-pixels", type=int, default=320 * 240, help="pixel budget per frame")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--model-dir", default=MODEL_DIR, help="local mode: model folder or Hugging Face id")
    ap.add_argument("--precision", choices=["4bit", "8bit", "bf16"], default="4bit",
                    help="local mode: 4bit for small GPUs, bf16 for 16 GB+ GPUs")
    ap.add_argument("--api", action="store_true", help="use Alibaba Cloud Model Studio instead of the local GPU")
    ap.add_argument("--send", choices=["auto", "video", "frames"], default="auto",
                    help="API only: upload the whole video file, or sampled frames (auto picks by file size)")
    ap.add_argument("--model", default=API_MODEL, help="API only: Model Studio model name")
    args = ap.parse_args()

    args.system = None
    if args.system_file:
        with open(args.system_file, encoding="utf-8") as f:
            args.system = f.read().strip()
    if args.prompt is None:
        if args.system:
            args.prompt = THINK_PROMPT if args.think else DEFAULT_PROMPT_WITH_SYSTEM
        else:
            args.prompt = DEFAULT_PROMPT

    jobs = []
    for v in args.videos:
        if not os.path.isfile(v):
            sys.exit(f"Video not found: {v}")
        jobs.append((os.path.basename(v), prepare_video(v, args.crop)))

    run_api(args, jobs) if args.api else run_local(args, jobs)


if __name__ == "__main__":
    main()
