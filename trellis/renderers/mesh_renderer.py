import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from ..representations.mesh import MeshExtractResult
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer as P3DMeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.ops.interp_face_attrs import (
    interpolate_face_attributes, #the cuda version
    interpolate_face_attributes_python, #the python version (keep here as a reminder)
)

def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix
    Args:
        intrinsics: [3, 3] OpenCV intrinsics matrix
        near: near plane to clip
        far: far plane to clip
    Returns:
        [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret

class MeshRenderer:
    def __init__(self, rendering_options=None, device='cuda'):
        """Initialize renderer with given options
        
        Args:
            rendering_options: Dictionary of rendering settings
            device: Device to use for rendering
        """
        self.rendering_options = edict({
            "resolution": None,
            "near": 0.1,
            "far": 1000.0,
            "ssaa": 1,
            "blur_radius": 0.0
        })
        if rendering_options is not None:
            self.rendering_options.update(rendering_options)
        
        self.device = device
        
        self.raster_settings = RasterizationSettings(
            image_size=self.rendering_options.resolution,
            blur_radius=self.rendering_options.blur_radius,
            faces_per_pixel=1,
            perspective_correct=True,
            clip_barycentric_coords=True,
            cull_backfaces=True,
            max_faces_per_bin=None,
            bin_size=None
        )
        
        self.rasterizer = MeshRasterizer(
            raster_settings=self.raster_settings
        )
        
        self.shader = SoftPhongShader(
            device=device,
            lights=None,
            cameras=None
        )
    
    def antialias(self, img: torch.Tensor, fragments) -> torch.Tensor:
        """Memory-efficient antialiasing with 2D chunking"""
        weights = fragments.bary_coords[..., 0]
        result = torch.empty_like(img)
        
        chunk_size = min(256, img.shape[1], img.shape[2])
        for i in range(0, img.shape[1], chunk_size):
            for j in range(0, img.shape[2], chunk_size):
                h_slice = slice(i, min(i + chunk_size, img.shape[1]))
                w_slice = slice(j, min(j + chunk_size, img.shape[2]))
                
                result[:, h_slice, w_slice] = (
                    img[:, h_slice, w_slice] * 
                    weights[:, h_slice, w_slice].unsqueeze(-1)
                ).sum(dim=-2)
        
        return result

    def _compute_vertex_normals(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        vertices = vertices if vertices.dim() == 3 else vertices.unsqueeze(0)
        faces = faces if faces.dim() == 3 else faces.unsqueeze(0)
        
        v0, v1, v2 = [vertices[:, faces[..., i]] for i in range(3)]
        face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0), dim=-1)
        face_normals = face_normals.squeeze(0).to(vertices.dtype)  # Match dtype
        
        vertex_normals = torch.zeros_like(vertices)
        for i in range(3):   #use faces[0, :, i] to get 1D index tensor:
            vertex_normals.index_add_(1, faces[0, :, i], face_normals)
        
        return F.normalize(vertex_normals, dim=-1)
    

    @torch.cuda.amp.autocast()
    def render(
            self,
            mesh: MeshExtractResult,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            return_types=["mask", "normal", "depth"]
        ) -> edict:
        """
        Render mesh using PyTorch3D
        
        Args:
            mesh: Mesh model to render
            extrinsics: [4, 4] camera extrinsics matrix
            intrinsics: [3, 3] camera intrinsics matrix
            return_types: List of outputs to generate. Valid types are:
                         "mask", "depth", "normal_map", "normal", "color"
                         
        Returns:
            Dictionary containing requested render outputs
        """
        if not all(t in ["mask", "normal", "depth", "normal_map", "color"] 
                  for t in return_types):
            raise ValueError(f"Invalid return type in {return_types}")
            
        torch.cuda.empty_cache()
        
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        
        # Handle empty mesh
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            default_img = torch.zeros((1, resolution, resolution, 3), 
                                   dtype=torch.float32, device=self.device)
            return edict({k: default_img if k in ['normal', 'normal_map', 'color'] 
                        else default_img[..., :1] for k in return_types})
        
        # Update rasterization settings
        self.raster_settings.image_size = resolution * ssaa
        
        # Camera setup
        perspective = intrinsics_to_projection(intrinsics, near, far)
        RT = extrinsics.unsqueeze(0)
        full_proj = (perspective @ extrinsics).unsqueeze(0)
        
        # Convert to PyTorch3D coordinates
        RT = RT.clone()
        RT[:, :3, 1:3] *= -1
        
        # Create cameras
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        cameras = PerspectiveCameras(
            focal_length=((2*fx, 2*fy),),
            principal_point=((2*cx-1, -2*cy+1),),
            R=RT[:, :3, :3],
            T=RT[:, :3, 3],
            device=self.device
        )
        
        # Prepare mesh
        vertices = mesh.vertices.unsqueeze(0).to(torch.float32)
        faces = mesh.faces.int()
        
        base_mesh = Meshes(
            verts=vertices,
            faces=faces.unsqueeze(0),
            textures=TexturesVertex(mesh.vertex_attrs.unsqueeze(0))
        )
        
        # Rasterize
        fragments = self.rasterizer(base_mesh, cameras=cameras)
        
        out_dict = edict()
        for type in return_types:
            if type == "mask":
                img = (fragments.zbuf > 0).float()
                img = self.antialias(img, fragments)
            
            elif type == "depth":
                depth = fragments.zbuf
                depth = (depth - near) / (far - near)
                img = torch.where(depth > 0, depth, 
                                torch.full_like(depth, float('inf')))
                img = self.antialias(img, fragments)
            
            elif type == "normal":
                # Compute smooth vertex normals
                vertex_normals = self._compute_vertex_normals(vertices, faces).to(torch.float32)#has to be float32 for the c++ code to work in pytorch32

                normal_mesh = Meshes(
                    verts=vertices,
                    faces=faces.unsqueeze(0),
                    textures=TexturesVertex(vertex_normals)
                )
                
                self.shader.cameras = cameras
                renderer = P3DMeshRenderer(
                    rasterizer=self.rasterizer,
                    shader=self.shader
                )
                img = renderer(normal_mesh, cameras=cameras)[..., :3]
                img = (img + 1) / 2
            
            elif type in ["normal_map", "color"]:
                attrs = (mesh.vertex_attrs[:, 3:] if type == "normal_map" 
                        else mesh.vertex_attrs[:, :3])
                face_attrs = attrs[faces]
                img = interpolate_face_attributes(
                    fragments.pix_to_face,
                    fragments.bary_coords,
                    face_attrs.unsqueeze(0)
                )
                img = self.antialias(img, fragments)
            
            # Handle supersampling
            if ssaa > 1:
                img = F.interpolate(
                    img.permute(0, 3, 1, 2),
                    (resolution, resolution),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True
                )
                img = img.squeeze()
            else:
                img = img.permute(0, 3, 1, 2).squeeze()
            
            out_dict[type] = img
        
        return out_dict