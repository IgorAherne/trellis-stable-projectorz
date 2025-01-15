import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, 
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.ops import vertex_normals

@dataclass
class RenderConfig:
    image_size: Tuple[int, int] = (512, 512)
    blur_radius: float = 1e-4
    faces_per_pixel: int = 1
    near: float = 0.1
    far: float = 1000.0
    max_batch_size: int = 4
    gradient_scale: float = 1.0
    perspective_correct: bool = True
    clip_barycentric_coords: bool = True
    cull_backfaces: bool = True

class Renderer:
    def __init__(self, 
                 device: str = "cuda",
                 config: Optional[RenderConfig] = None):
        self.device = device
        self.config = config or RenderConfig()
        
        self.raster_settings = RasterizationSettings(
            image_size=max(self.config.image_size),
            blur_radius=self.config.blur_radius,
            faces_per_pixel=self.config.faces_per_pixel,
            perspective_correct=self.config.perspective_correct,
            clip_barycentric_coords=self.config.clip_barycentric_coords,
            cull_backfaces=self.config.cull_backfaces
        )
        
        self.rasterizer = MeshRasterizer(
            raster_settings=self.raster_settings
        )
        
        self._setup_shaders()

    def _setup_shaders(self):
        """Initialize specialized shaders for different render passes"""
        self.normal_shader = SoftPhongShader(
            device=self.device,
            lights=None,
            cameras=None  # Will be set per render
        )

    def _create_cameras(self, 
                       mv: torch.Tensor, 
                       proj: torch.Tensor, 
                       image_size: Tuple[int, int]) -> PerspectiveCameras:
        """Convert OpenGL-style camera matrices to PyTorch3D camera parameters"""
        mv = mv.clone()
        mv[:, :3, 1:3] *= -1  # OpenGL to PyTorch3D coordinate system
        
        # Extract camera parameters from projection matrix
        focal_length = 2.0 / proj[..., 1, 1]
        principal_point = torch.zeros_like(focal_length)[..., None].repeat(1, 2)
        
        return PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=mv[:, :3, :3],
            T=mv[:, :3, 3],
            device=self.device,
            in_ndc=True,
            image_size=image_size
        )

    def _prepare_mesh(self, 
                     vertices: torch.Tensor, 
                     faces: torch.Tensor,
                     vertex_colors: Optional[torch.Tensor] = None) -> Meshes:
        """Convert raw vertices and faces to PyTorch3D mesh structure"""
        vertices = vertices if vertices.dim() == 3 else vertices.unsqueeze(0)
        faces = faces if faces.dim() == 3 else faces.unsqueeze(0)
        
        if vertex_colors is None:
            vertex_colors = torch.ones_like(vertices)
        else:
            vertex_colors = vertex_colors if vertex_colors.dim() == 3 else vertex_colors.unsqueeze(0)
            
        return Meshes(
            verts=vertices * self.config.gradient_scale,
            faces=faces,
            textures=TexturesVertex(vertex_colors)
        )

    @torch.cuda.amp.autocast()
    def render_mesh(self, 
                   vertices: torch.Tensor, 
                   faces: torch.Tensor,
                   mv: torch.Tensor, 
                   mvp: torch.Tensor, 
                   image_size: Tuple[int, int],
                   return_types: List[str] = ["mask", "depth", "normal"],
                   white_bg: bool = False,
                   vertex_colors: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Main render function supporting masks, depth maps, and normal maps"""
        torch.cuda.empty_cache()
        self.rasterizer.raster_settings.image_size = image_size
        
        mesh = self._prepare_mesh(vertices, faces, vertex_colors)
        cameras = self._create_cameras(mv, mvp, image_size)
        fragments = self.rasterizer(mesh, cameras=cameras)
        
        outputs = {}
        if "mask" in return_types:
            outputs["mask"] = (fragments.zbuf > 0).float()

        if "depth" in return_types:
            depth = (fragments.zbuf - self.config.near) / (self.config.far - self.config.near)
            outputs["depth"] = torch.where(
                depth > 0, 
                depth, 
                torch.full_like(depth, float('inf'))
            )

        if "normal" in return_types:
            vert_normals = vertex_normals(vertices, faces)
            normal_mesh = Meshes(
                verts=vertices,
                faces=faces,
                textures=TexturesVertex(vert_normals)
            )
            
            self.normal_shader.cameras = cameras
            normal_images = MeshRenderer(
                rasterizer=self.rasterizer,
                shader=self.normal_shader
            )(normal_mesh, cameras=cameras)
            
            outputs["normal"] = normal_images[..., :3] * 2 - 1

        if white_bg:
            mask = outputs["mask"]
            outputs.update({
                k: v * mask + (1 - mask) if k != "depth" else v 
                for k, v in outputs.items()
            })

        return outputs

def perspective(fovy: float, 
               aspect: float = 1.0, 
               near: float = 0.1, 
               far: float = 1000.0, 
               device: str = "cuda") -> torch.Tensor:
    """Create OpenGL-style perspective projection matrix"""
    y = torch.tensor(np.tan(fovy / 2), device=device)
    return torch.tensor([
        [1/(y*aspect), 0, 0, 0],
        [0, -1/y, 0, 0],
        [0, 0, -(far+near)/(far-near), -(2*far*near)/(far-near)],
        [0, 0, -1, 0]
    ], device=device)

def get_rotate_camera(angle: float, 
                     radius: float = 3.0, 
                     fov: float = 0.5,
                     res: Tuple[int, int] = (512, 512), 
                     device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate camera matrices for rotating view"""
    mv = torch.eye(4, device=device)
    mv[2, 3] = radius
    
    rot = torch.tensor([
        [np.cos(angle), 0, -np.sin(angle), 0],
        [0, 1, 0, 0],
        [np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]
    ], device=device)
    
    mv = mv @ rot
    proj = perspective(fov, res[1]/res[0], 1, 50, device)
    return mv, proj @ mv

def get_random_camera_batch(batch_size: int, 
                          radius: float = 3.0,
                          fov_min: float = 0.3, 
                          fov_max: float = 0.7,
                          res: Tuple[int, int] = (512, 512),
                          device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate batch of random camera views"""
    mv = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    mv[:, 2, 3] = radius
    
    # Random spherical coordinates
    theta = torch.rand(batch_size, device=device) * 2 * np.pi
    phi = torch.arccos(torch.rand(batch_size, device=device) * 2 - 1)
    
    # Create rotation matrices
    Ry = torch.stack([
        torch.cos(theta), torch.zeros_like(theta), -torch.sin(theta),
        torch.zeros_like(theta), torch.ones_like(theta), torch.zeros_like(theta),
        torch.sin(theta), torch.zeros_like(theta), torch.cos(theta)
    ], dim=1).reshape(-1, 3, 3)
    
    Rx = torch.stack([
        torch.ones_like(phi), torch.zeros_like(phi), torch.zeros_like(phi),
        torch.zeros_like(phi), torch.cos(phi), -torch.sin(phi),
        torch.zeros_like(phi), torch.sin(phi), torch.cos(phi)
    ], dim=1).reshape(-1, 3, 3)
    
    mv[:, :3, :3] = torch.bmm(Ry, Rx)
    
    # Random FOV between min and max
    fov = torch.rand(batch_size, device=device) * (fov_max - fov_min) + fov_min
    proj = torch.stack([
        perspective(f, res[1]/res[0], 1, 50, device) 
        for f in fov
    ])
    
    return mv, torch.bmm(proj, mv)