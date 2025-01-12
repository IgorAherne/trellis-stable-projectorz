from enum import Enum
from typing import Optional, Dict
from fastapi import Form
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PROCESSING = "PROCESSING"
    PREVIEW_READY = "PREVIEW_READY"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class GenerationArgForm:
    def __init__(
        self,
        seed: int = Form(1),
        prompt_positive: str = Form(""),
        ss_guidance_strength: float = Form(7.5),
        ss_sampling_steps: int = Form(12),
        slat_guidance_strength: float = Form(3.0),
        slat_sampling_steps: int = Form(12),
        preview_resolution: int = Form(512),
        preview_frames: int = Form(150),
        preview_fps: int = Form(20),
        mesh_simplify_ratio: float = Form(0.95), 
        texture_size: int = Form(1024),
        remove_background:bool = Form(True),
        output_format: str = Form("glb"),
    ):
        self.seed = seed
        self.prompt_positive = prompt_positive
        self.ss_guidance_strength = ss_guidance_strength
        self.ss_sampling_steps = ss_sampling_steps
        self.slat_guidance_strength = slat_guidance_strength
        self.slat_sampling_steps = slat_sampling_steps
        self.preview_resolution = preview_resolution
        self.preview_frames = preview_frames
        self.preview_fps = preview_fps
        self.mesh_simplify_ratio = mesh_simplify_ratio
        self.texture_size = texture_size
        self.remove_background = remove_background
        self.output_format = output_format



class GenerationResponse(BaseModel):
    # No task_id anymore, we focus on a single generation
    status: TaskStatus
    progress: int = 0
    message: str = ""
    # Only used if we did “generate_preview”
    preview_urls: Optional[Dict[str, str]] = None
    # Only used if generation is complete
    model_url: Optional[str] = None


class StatusResponse(BaseModel):
    status: TaskStatus
    progress: int
    message: str
    busy: bool
