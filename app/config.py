from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CASH_", env_file=".env", extra="ignore"
    )
    api_key: str = ""
    ui_password: str = ""
    ui_user: str = "reviewer"
    data_dir: Path = Path("runtime")
    frontend_dir: Path = Path("frontend/dist")
    model_enabled: bool = False
    engine: Literal["qwen", "nmk"] = "nmk"
    nmk_dir: Path = Path("nmk")
    nmk_model_path: Path = Path("/root/models/Qwen3-VL-8B-Instruct")
    nmk_precision: Literal["4bit", "8bit", "bf16"] = "bf16"
    ui_open: bool = False
    model_path: Path = Path("models/Qwen3-VL-4B-Instruct")
    lora_path: str = ""
    device: str = "cuda:0"
    qwen_input_mode: Literal["images", "video"] = "images"
    qwen_discovery_style: Literal["narrative", "checklist"] = "narrative"
    calibration_profile_path: str = ""
    job_timeout_sec: float = Field(290, gt=0, le=300)
    queue_capacity: int = Field(3, ge=1, le=50)
    max_upload_mb: int = Field(100, ge=1, le=500)
    overview_fps: float = Field(1, gt=0, le=5)
    max_overview_frames: int = Field(145, ge=2, le=300)
    max_detail_frames: int = Field(24, ge=2, le=100)
    context_frame_count: int = Field(24, ge=2, le=48)
    audit_frame_count: int = Field(32, ge=2, le=64)
    detail_frame_max_pixels: int = Field(524288, ge=4096, le=1048576)
    max_candidates: int = Field(6, ge=1, le=20)
    max_visual_tokens: int = Field(20000, ge=256)
    max_input_tokens: int = Field(28000, ge=512)
    frame_max_pixels: int = Field(131072, ge=4096, le=1048576)

    @model_validator(mode="after")
    def secrets(self):
        if len(self.api_key) < 24 or len(self.ui_password) < 24:
            raise ValueError(
                "Set CASH_API_KEY and CASH_UI_PASSWORD to distinct secrets of at least 24 characters"
            )
        if self.api_key == self.ui_password:
            raise ValueError("API key and UI password must be different")
        return self
