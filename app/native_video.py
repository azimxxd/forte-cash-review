"""Native Qwen3-VL temporal input from locally decoded frames and their real PTS."""

import math
import numpy as np


def prepare_native_video(processor, frames, messages, max_visual_tokens, source_fps):
    if len(frames) < 2 or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(
            "Native video needs timestamped frames and a positive source FPS"
        )
    timestamps = [frame.timestamp for frame in frames]
    if timestamps != sorted(timestamps):
        raise ValueError("Native frames must be chronological")
    images = [frame.image() for frame in frames]
    if len({image.size for image in images}) != 1:
        raise ValueError("Native video frames must share dimensions")
    video = np.stack([np.asarray(image) for image in images])
    indices = [round(timestamp * source_fps) for timestamp in timestamps]
    metadata = {
        "fps": source_fps,
        "frames_indices": indices,
        "total_num_frames": max(indices) + 1,
        "duration": timestamps[-1] + 1 / source_fps,
        "width": images[0].width,
        "height": images[0].height,
        "video_backend": "local_pyav",
    }
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    pixel_factor = (
        processor.video_processor.patch_size * processor.video_processor.merge_size
    ) ** 2
    temporal = processor.video_processor.temporal_patch_size
    inputs = processor(
        text=[text],
        videos=[video],
        return_tensors="pt",
        videos_kwargs={
            "do_sample_frames": False,
            "video_metadata": [metadata],
            "size": {
                "shortest_edge": 4096 * temporal,
                "longest_edge": int(max_visual_tokens * 0.85) * pixel_factor * temporal,
            },
        },
    )
    grid = inputs.get("video_grid_thw")
    if (
        grid is None
        or grid.shape[0] != 1
        or int(grid[0, 0]) * temporal not in (len(frames), len(frames) + 1)
    ):
        raise ValueError("Native processor changed or omitted the supplied frames")
    tokens = (
        int(grid.prod(dim=1).sum().item()) // processor.video_processor.merge_size**2
    )
    if tokens > max_visual_tokens:
        raise ValueError("Native video exceeds visual token budget")
    return inputs, tokens
