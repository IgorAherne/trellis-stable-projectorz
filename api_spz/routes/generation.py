import gc
import json
import logging
import threading
import time
import traceback
from typing import Dict, Optional, Literal, List, Union
import asyncio
import io
import base64
import os
from pathlib import Path
from fastapi import APIRouter, File, Response, UploadFile, Form, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from PIL import Image
import torch
import imageio

from api_spz.core.exceptions import CancelledException
from api_spz.core.files_manage import file_manager
from api_spz.core.state_manage import state
from api_spz.core.models_pydantic import (
    GenerationArgForm,
    GenerationResponse, 
    TaskStatus,
    StatusResponse
)

# Trellis pipeline + utils
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


router = APIRouter()

logger = logging.getLogger("trellis") #was already setup earlier, during main.

cancel_event = asyncio.Event() # This event will be set by the endpoint /interrupt

# A single lock to ensure only one generation at a time
generation_lock = asyncio.Lock()
def is_generation_in_progress() -> bool:
    return generation_lock.locked()


# A single dictionary holding "current generation" metadata
current_generation = {
    "status": TaskStatus.FAILED, # default
    "progress": 0,
    "message": "",
    "outputs": None,       # pipeline outputs if we did partial gen.
    "preview_urls": None,  # dict of preview paths if relevant.
    "model_url": None      # final model path if relevant.
}


# Helper to reset the "current_generation" dictionary
# (useful to start fresh each time we begin generating)
def reset_current_generation():
    cancel_event.clear()
    current_generation["status"] = TaskStatus.PROCESSING
    current_generation["progress"] = 0
    current_generation["message"] = ""
    current_generation["outputs"] = None
    current_generation["preview_urls"] = None
    current_generation["model_url"] = None


# Helper to update the "current_generation" dictionary
def update_current_generation(
    status: Optional[TaskStatus] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    outputs=None
):
    if status is not None:
        current_generation["status"] = status
    if progress is not None:
        current_generation["progress"] = progress
    if message is not None:
        current_generation["message"] = message
    if outputs is not None:
        current_generation["outputs"] = outputs


# Cleanup files in "current_generation" folder
async def cleanup_generation_files(keep_videos: bool = False, keep_model: bool = False):
    file_manager.cleanup_generation_files(keep_videos=keep_videos, keep_model=keep_model)


# Validate input
def _gen_3d_validate_params(file_or_files, b64_or_b64list, arg: GenerationArgForm):
    """Validate incoming parameters before generation."""
    if (not file_or_files or len(file_or_files) == 0) and (not b64_or_b64list or len(b64_or_b64list) == 0):
        raise HTTPException(400, "No input images provided")
    # Range checks:
    if not (0 < arg.ss_guidance_strength <= 10):
        raise HTTPException(status_code=400, detail="SS guidance strength must be above 0 and <= 10")
    if not (0 < arg.ss_sampling_steps <= 50):
        raise HTTPException(status_code=400, detail="SS sampling steps must be above 0 and <= 50")
    if not (0 < arg.slat_sampling_steps <= 50):
        raise HTTPException(status_code=400, detail="Slat sampling steps must be above 0 and <= 50")
    if not (0 < arg.slat_guidance_strength <= 10):
        raise HTTPException(status_code=400, detail="Slat guidance strength must be above 0 and <= 10")
    if not (15 < arg.preview_frames <= 1000):
        raise HTTPException(status_code=400, detail="Preview frames must be above 15 and <= 1000")
    if not (0 <= arg.mesh_simplify_ratio <= 100):  # Change upper bound to 100
        raise HTTPException(status_code=400, detail="mesh_simplify_ratio must be between 0 and 1, or between 0 and 100")

    if arg.output_format not in ["glb", "gltf"]:
        raise HTTPException(status_code=400, detail="Unsupported output format")



async def _gen_3d_get_image(file: Optional[UploadFile], image_base64: Optional[str]) -> Image.Image:
    if image_base64:
        try:
            # base64 branch
            if "base64," in image_base64:
                image_base64 = image_base64.split("base64,")[1]
            data = base64.b64decode(image_base64)
            pil_image = Image.open(io.BytesIO(data)).convert("RGBA")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 data: {str(e)}")
    else:
        try:
            content = await file.read()
            pil_image = Image.open(io.BytesIO(content)).convert("RGBA")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"get_image - issue opening the image file: {str(e)}")
    return pil_image



# Reads multiple images from either UploadFile objects or base64 strings,
# and returns a list of PIL.Image.
async def _load_images_into_list( files: Optional[List[UploadFile]] = None,
                                  images_base64: Optional[List[str]] = None) -> List[Image.Image]:
    all_images = []
    files = files or []
    images_base64 = images_base64 or []
    # 1) Base64-encoded:
    for b64_str in images_base64:
        if "base64," in b64_str:
            b64_str = b64_str.split("base64,")[1]
        img = await _gen_3d_get_image(file=None, image_base64=b64_str)
        all_images.append(img)
    # 2) Files:
    for f in files:
        img = await _gen_3d_get_image(file=f, image_base64=None)
        all_images.append(img)

    if not all_images:
        raise HTTPException(status_code=400, detail="No images provided (files or base64).")

    return all_images



# If `pil_images` is a single PIL.Image, calls pipeline.run(...).
# If `pil_images` is a list of multiple PIL.Images, calls pipeline.run_multi_image(...).
# Returns the pipeline outputs dictionary.
async def _run_pipeline_generate_3d( pil_images: Union[Image.Image, List[Image.Image]],  arg):
    def worker():
        sparse_structure_sampler_params={
            "steps": arg.ss_sampling_steps,
            "cfg_strength": arg.ss_guidance_strength,
        }
        slat_sampler_params={
            "steps": arg.slat_sampling_steps,
            "cfg_strength": arg.slat_guidance_strength,
        }
        pipeline = state.pipeline
        torch.cuda.empty_cache()
        gc.collect()
        # Decide single-image vs multi-image
        if isinstance(pil_images, list):
            if len(pil_images) > 1:# Multi-image scenario:
                outputs = pipeline.run_multi_image(
                    pil_images,
                    seed=arg.seed,
                    sparse_structure_sampler_params = sparse_structure_sampler_params,
                    slat_sampler_params = slat_sampler_params,
                    formats = ['mesh', 'gaussian',],
                    cancel_event=cancel_event )
            else: # Single PIL image in a list
                outputs = pipeline.run(
                    pil_images[0],
                    seed=arg.seed,
                    sparse_structure_sampler_params = sparse_structure_sampler_params,
                    slat_sampler_params = slat_sampler_params,
                    formats = ['mesh', 'gaussian',],
                    cancel_event=cancel_event )
        else:# It's truly a single PIL.Image:
            outputs = pipeline.run(
                pil_images,
                seed=arg.seed,
                sparse_structure_sampler_params = sparse_structure_sampler_params,
                slat_sampler_params = slat_sampler_params,
                formats = ['mesh', 'gaussian',],
                cancel_event=cancel_event )
        torch.cuda.empty_cache()
        gc.collect()
        return outputs
        # end worker()
    outputs = await asyncio.to_thread(worker)
    return outputs



async def _run_pipeline_generate_previews(outputs, resolution:int, preview_frames:int, preview_fps:int):
    """Generate the preview videos in a thread, saves them to disk."""
    def worker():
        torch.cuda.empty_cache()
        gc.collect()
        videos = {
            "gaussian": render_utils.render_video(
                outputs["gaussian"][0],
                resolution=resolution,
                num_frames=preview_frames,
                cancel_event=cancel_event
            )["color"],
            # For performance reasons, skipping these videos:
            # "mesh": render_utils.render_video(
            #     outputs["mesh"][0],
            #     resolution=resolution,
            #     num_frames=preview_frames,
            #     cancel_event=cancel_event
            # )["normal"],
            # "radiance": render_utils.render_video(
            #     outputs["radiance_field"][0],
            #     resolution=resolution,
            #     num_frames=preview_frames,
            #     cancel_event=cancel_event
            # )["color"]
        }
        for name, video in videos.items():
            preview_path = file_manager.get_temp_path(f"preview_{name}.mp4")
            #use the video settings that unity3D can work with:
            imageio.mimsave(str(preview_path),  video,  fps=preview_fps,  codec="libx264",  format="mp4", 
                            pixelformat="yuv420p",  ffmpeg_params=["-profile:v", "baseline", "-level", "3.0"])
            torch.cuda.empty_cache()
            gc.collect()

    await asyncio.to_thread(worker)


# Before using mesh_simplify_ratio in pipeline calls:
def normalize_meshSimplify_ratio(ratio: float) -> float:
    """Normalize mesh_simplify_ratio to [0,1] range"""
    if ratio > 0.999:  # Detect [0,100] range
        return ratio / 100.0
    return ratio


async def _run_pipeline_generate_glb(outputs, mesh_simplify_ratio: float, texture_size: int):
    """Generate the final GLB model in a thread."""
    mesh_simplify_ratio = normalize_meshSimplify_ratio(mesh_simplify_ratio)

    def worker():
        torch.cuda.empty_cache()
        gc.collect()
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=mesh_simplify_ratio,
            texture_size=texture_size,
            cancel_event=cancel_event
        )
        model_path = file_manager.get_temp_path("model.glb")
        glb.export(str(model_path))
        torch.cuda.empty_cache()
        gc.collect()

    await asyncio.to_thread(worker)

# --------------------------------------------------
# Routes
# --------------------------------------------------

@router.get("/ping")
async def ping():
    """Root endpoint to check server status."""
    busy = is_generation_in_progress()
    return {
        "status": "running",
        "message": "Trellis API is operational",
        "busy": busy
    }


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get status of the single current/last generation.
    """
    return StatusResponse(
        status=current_generation["status"],
        progress=current_generation["progress"],
        message=current_generation["message"],
        busy=is_generation_in_progress(),
    )


@router.post("/generate_no_preview", response_model=GenerationResponse)
async def generate_no_preview(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    arg: GenerationArgForm = Depends()
):
    """Generate a 3D model directly (no preview)."""
    print()#empty line, for an easier read.
    logger.info("Client asked to generate with no previews")
    # Acquire the lock (non-blocking)
    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException( status_code=503, detail="Server is busy with another generation")
    
    start_time = time.time() 
    # We have the lock => let's reset the "current_generation"
    reset_current_generation()
    try:
        _gen_3d_validate_params(file, image_base64, arg)

        #construct an image
        image = await _gen_3d_get_image(file, image_base64)

        update_current_generation( status=TaskStatus.PROCESSING, progress=10, message="Generating 3D structure...")
        outputs = await _run_pipeline_generate_3d(image, arg)
        update_current_generation( progress=50, message="3D structure generated", outputs=outputs)

        # Generate final GLB
        update_current_generation( progress=70, message="Generating GLB file..." )
        await _run_pipeline_generate_glb(outputs, arg.mesh_simplify_ratio, arg.texture_size)

        # Done
        update_current_generation( status=TaskStatus.COMPLETE, progress=100, message="Generation complete")

        # Clean up intermediate files, keep final model
        await cleanup_generation_files(keep_model=True)

        duration = time.time() - start_time  # Calculate duration before return
        logger.info(f"Generation completed in {duration:.2f} seconds")
        return GenerationResponse(
            status=TaskStatus.COMPLETE,
            progress=100,
            message="Generation complete",
            model_url="/download/model"  # single endpoint
        )
    except CancelledException as cex:
        # Cancel was triggered mid-generation
        update_current_generation( status=TaskStatus.FAILED, progress=0, message="Cancelled by user")
        await cleanup_generation_files()
        raise HTTPException(status_code=499, detail="Generation cancelled by user")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        update_current_generation( status=TaskStatus.FAILED, progress=0, message=str(e))
        await cleanup_generation_files()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()


@router.post("/generate_preview", response_model=GenerationResponse)
async def generate_preview(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    arg: GenerationArgForm = Depends(),
):
    """ Generate partial 3D structure + Previews, let user resume with /resume_from_preview """
    print()#empty line, for an easier read.
    logger.info("Client asked to generate with previews")
    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server is busy with another generation" )

    reset_current_generation()

    try:
        _gen_3d_validate_params(file, image_base64, arg)

        #construct an image
        image = await _gen_3d_get_image(file, image_base64)

        update_current_generation(status=TaskStatus.PROCESSING, progress=20, message="Generating 3D structure...")
        outputs = await _run_pipeline_generate_3d(image, arg)
        update_current_generation(progress=50, message="3D structure generated", outputs=outputs)

        # Generate Previews
        update_current_generation( progress=60, message="Generating previews..." )
        await _run_pipeline_generate_previews(outputs, arg.preview_resolution, arg.preview_frames, arg.preview_fps)

        # Set up preview URLs
        preview_urls = {
            "gaussian": "/download/preview/gaussian",
            "mesh": "/download/preview/mesh",
            #we don't need radiance to make mesh+texture, so omitting for performance reasons:
            #   "radiance": "/download/preview/radiance",  
        }
        update_current_generation(status=TaskStatus.PREVIEW_READY, progress=100, message="Preview generation complete")
        current_generation["preview_urls"] = preview_urls

        # Clean up everything except the preview videos
        await cleanup_generation_files(keep_videos=True)

        return GenerationResponse(
            status=TaskStatus.PREVIEW_READY,
            progress=100,
            message="Preview generation complete",
            preview_urls=preview_urls
        )
    except CancelledException as cex:
        # Cancel was triggered mid-generation
        update_current_generation( status=TaskStatus.FAILED, progress=0, message="Cancelled by user")
        await cleanup_generation_files()
        raise HTTPException(status_code=499, detail="Generation cancelled by user")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        update_current_generation(status=TaskStatus.FAILED, progress=0, message=str(e))
        await cleanup_generation_files()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()


@router.post("/generate_multi_no_preview", response_model=GenerationResponse)
async def generate_multi_no_preview(
    file_list: Optional[List[UploadFile]] = File(None),
    image_list_base64: Optional[List[str]] = Form(None),
    arg: GenerationArgForm = Depends(),
):
    """
    Generate a 3D model using multiple images, directly (no preview).
    The pipeline will receive [img1, img2, ...] as input.
    """
    print()#empty line, for an easier read.
    logger.info("Client asked to multi-generate with no previews")
    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException( status_code=503, detail="Server is busy with another generation")
    
    start_time = time.time() 
    # We have the lock => let's reset the "current_generation"
    reset_current_generation()

    try:
        _gen_3d_validate_params(file_list, image_list_base64, arg)
        # construct images:
        images = await _load_images_into_list(file_list, image_list_base64)

        update_current_generation(status=TaskStatus.PROCESSING, progress=10, message="Generating 3D structure...")
        outputs = await _run_pipeline_generate_3d(images, arg)
        update_current_generation(progress=50, message="3D structure generated", outputs=outputs)

        # 2) Generate final GLB
        await _run_pipeline_generate_glb(outputs, arg.mesh_simplify_ratio, arg.texture_size)

        # Done
        update_current_generation(status=TaskStatus.COMPLETE, progress=100, message="Generation complete")

        # Clean up intermediate files, keep final model
        await cleanup_generation_files(keep_model=True)

        duration = time.time() - start_time  # Calculate duration before return
        logger.info(f"Generation completed in {duration:.2f} seconds")
        return GenerationResponse(
            status=TaskStatus.COMPLETE,
            progress=100,
            message="Generation complete",
            model_url="/download/model"  # single endpoint
        )
    except CancelledException as cex:
        # Cancel was triggered mid-generation
        update_current_generation( status=TaskStatus.FAILED, progress=0, message="Cancelled by user")
        await cleanup_generation_files()
        raise HTTPException(status_code=499, detail="Generation cancelled by user")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        update_current_generation(status=TaskStatus.FAILED, progress=0, message=str(e))
        await cleanup_generation_files()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()



@router.post("/generate_multi_preview", response_model=GenerationResponse)
async def generate_multi_preview(
    file_list: Optional[List[UploadFile]] = File(None),
    image_list_base64: Optional[List[str]] = Form(None),
    arg: GenerationArgForm = Depends(),
):
    """
    Generate partial 3D structure + Previews using multiple images.
    Let user resume with /resume_from_preview
    """
    print()#empty line, for an easier read.
    logger.info("Client asked to multi-generate with previews")
    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server is busy with another generation")

    reset_current_generation()

    try:
        if arg.output_format not in ["glb", "gltf"]:
            raise HTTPException(status_code=400, detail="Unsupported output format")
        # construct images:
        images = await _load_images_into_list(file_list, image_list_base64)

        update_current_generation(status=TaskStatus.PROCESSING, progress=20, message="Generating 3D structure...")
        outputs = await _run_pipeline_generate_3d(images, arg)
        update_current_generation(progress=50, message="3D structure generated", outputs=outputs)

        # Generate Previews (gaussian, mesh, radiance)
        update_current_generation(progress=60, message="Generating previews...")

        def worker_previews():
            videos = {
                "gaussian": render_utils.render_video(
                    outputs["gaussian"][0],
                    resolution=arg.preview_resolution,
                    num_frames=arg.preview_frames
                )["color"],
                #we don't need radiance to make mesh+texture, so omitting for performance reasons:
                #  "mesh": render_utils.render_video(
                #      outputs["mesh"][0],
                #      resolution=arg.preview_resolution,
                #      num_frames=arg.preview_frames
                #  )["normal"],
                #  "radiance": render_utils.render_video(
                #      outputs["radiance_field"][0],
                #      resolution=arg.preview_resolution,
                #      num_frames=arg.preview_frames
                #  )["color"],
            }
            for name, video in videos.items():
                preview_path = file_manager.get_temp_path(f"preview_{name}.mp4")
                imageio.mimsave(
                    str(preview_path),
                    video,
                    fps=arg.preview_fps,
                    codec="libx264",
                    format="mp4",
                    pixelformat="yuv420p",
                    ffmpeg_params=["-profile:v", "baseline", "-level", "3.0"]
                )
        await asyncio.to_thread(worker_previews)

        # Set up preview URLs
        preview_urls = {
            "gaussian": "/download/preview/gaussian",
            "mesh": "/download/preview/mesh",
            # we don't need radiance to make mesh+texture, so omitting for performance reasons:
            #   "radiance": "/download/preview/radiance",
        }
        update_current_generation(status=TaskStatus.PREVIEW_READY, progress=100, message="Preview generation complete")
        current_generation["preview_urls"] = preview_urls

        # Clean up everything except the preview videos
        await cleanup_generation_files(keep_videos=True)

        return GenerationResponse(
            status=TaskStatus.PREVIEW_READY,
            progress=100,
            message="Preview generation complete",
            preview_urls=preview_urls
        )
    except CancelledException as cex:
        # Cancel was triggered mid-generation
        update_current_generation( status=TaskStatus.FAILED, progress=0, message="Cancelled by user")
        await cleanup_generation_files()
        raise HTTPException(status_code=499, detail="Generation cancelled by user")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        update_current_generation(status=TaskStatus.FAILED, progress=0, message=str(e))
        await cleanup_generation_files()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()


# can be invoked after user generated previews and is happy with them.
# here we'll complete the generation and will make the mesh available for a download.
@router.post("/resume_from_preview", response_model=GenerationResponse)
async def resume_from_preview(
    mesh_simplify_ratio: float = Query(0.95, gt=0, le=1),
    texture_size: int = Query(1024, gt=0, le=4096),
):
    """Resume from a PREVIEW_READY state, generate final GLB."""
    print()#empty line, for an easier read.
    logger.info("Client resumed from a preview video.")
    if is_generation_in_progress():
        raise HTTPException(
            status_code=503,
            detail="Server is busy with another generation"
        )
    
    try:# Acquire lock right away
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.001)
    except:
        raise HTTPException(status_code=503, detail="Server is busy with another generation")

    try:
        if current_generation["status"] != TaskStatus.PREVIEW_READY:
            raise HTTPException(
                status_code=400,
                detail="Current generation must be in preview_ready state to resume"
            )
        outputs = current_generation["outputs"]
        if not outputs:
            raise HTTPException(
                status_code=400,
                detail="No pipeline outputs found in memory"
            )
        update_current_generation( status=TaskStatus.PROCESSING, progress=70, message="Generating final GLB...")
        await _run_pipeline_generate_glb(outputs, mesh_simplify_ratio, texture_size)

        update_current_generation( status=TaskStatus.COMPLETE, progress=100, message="Generation complete")
        await cleanup_generation_files(keep_model=True)# Cleanup everything except final model

        return GenerationResponse(
            status=TaskStatus.COMPLETE,
            progress=100,
            message="Generation complete",
            model_url="/download/model"
        )
    except CancelledException as cex:
        # Cancel was triggered mid-generation
        update_current_generation( status=TaskStatus.FAILED, progress=0, message="Cancelled by user")
        await cleanup_generation_files()
        raise HTTPException(status_code=499, detail="Generation cancelled by user")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        update_current_generation(
            status=TaskStatus.FAILED,
            progress=0,
            message=str(e)
        )
        await cleanup_generation_files()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()




@router.post("/generate", response_model=GenerationResponse)
async def process_ui_generation_request(
    data: Dict
):
    """Process generation request from the UI panel and redirect to appropriate endpoint."""

    try:
        # Extract values from simplified input format
        arg = GenerationArgForm(
            seed = int(data.get("seed", 1)),
            ss_guidance_strength = data.get("ss_strength", 7.5),
            ss_sampling_steps = int(data.get("ss_steps", 12)),
            slat_guidance_strength = data.get("slat_strength", 3.0),
            slat_sampling_steps = int(data.get("slat_steps", 12)),
            preview_resolution = 512,
            preview_frames     = 150,
            preview_fps        = 20,
            mesh_simplify_ratio = data.get("mesh_simplify", 0.95),
            texture_size = int(data.get("texture_size", 1024)),
            output_format = "glb"
        )
        # Get images from input
        images_base64 = data.get("single_multi_img_input", [])
        if not images_base64:
            raise HTTPException(status_code=400, detail="No images provided")

        # Decide whether to generate preview based on input
        skip_videos = data.get("skip_videos", True)

        # for now always skip videos (StableProjectorz doesn't show them) - 20 Jan 2025
        response = await generate_multi_no_preview(
            file_list=None,
            image_list_base64=images_base64,
            arg=arg
        )
        # if skip_videos:
        #     # Call generate_multi_no_preview
        #     response = await generate_multi_no_preview(
        #         file_list=None,
        #         image_list_base64=images_base64,
        #         arg=arg
        #     )
        # else:
        #     # Call generate_multi_preview
        #     response = await generate_multi_preview(
        #         file_list=None,
        #         image_list_base64=images_base64,
        #         arg=arg
        #     )
        return response

    except Exception as e:
        logger.error(f"Error processing UI generation request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



# for example:
#   "make_meshes_and_tex",
#   "retexture",
#   "retexture_via_masks"  etc (see api-documentation.html)
@router.get("/info/supported_operations")
async def get_supported_operation_types():
   return ["make_meshes_and_tex"]




@router.post("/interrupt")
async def interrupt_generation():
    """Interrupt the current generation if one is in progress."""
    logger.info("Client cancelled the generation.")
    if not is_generation_in_progress():
        return {"status": "no_generation_in_progress"}
    cancel_event.set()  # <-- Signal cancellation
    return {"status": "interrupt_requested"}
    


@router.get("/download/preview/{type}")
async def download_preview(
    type: Literal["gaussian", "mesh",] #"radiance"]  #we don't need radiance to make mesh+texture, so omitting for performance reasons.
):
    """Download the preview video for the current generation."""
    logger.info("Client is downloading a video.")
    preview_path = file_manager.get_temp_path(f"preview_{type}.mp4")
    if not preview_path.exists():
        logger.error(f"preview_{type}.mp4 not found")
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(
        str(preview_path),
        media_type="video/mp4",
        filename=f"preview_{type}.mp4"
    )


@router.get("/download/model")
async def download_model():
    """Download final 3D model (GLB)."""
    logger.info("Client is downloading a model.")
    model_path = file_manager.get_temp_path("model.glb")
    if not model_path.exists():
        logger.error(f"mesh not found")
        raise HTTPException(status_code=404, detail="Mesh not found")
    return FileResponse(
        str(model_path),
        media_type="model/gltf-binary",
        filename="model.glb"
    )


@router.get("/download/spz-ui-layout/generation-3d-panel")
async def get_generation_panel_layout():
    """Return the UI layout for the generation panel."""
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'layout_generation_3d_panel.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # By using Response(content=content, media_type="text/plain; charset=utf-8")
            # - Bypass the automatic JSON encoding
            # - Explicitly tell the client this is plain text (not JSON)
            # - Ensure proper UTF-8 encoding is maintained
            # This way Unity receives the layout text exactly as it appears in the file.
            # It keeps proper line breaks and formatting intact, with special characters not being escaped:
            return Response(content=content,  media_type="text/plain; charset=utf-8")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Layout file not found")