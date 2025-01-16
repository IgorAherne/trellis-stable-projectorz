import torch
import numpy as np
import torch.nn.functional as F

def do_texture(tex, uv, uv_da=None, mip_level_bias=None, mip=None, 
               filter_mode='auto', boundary_mode='wrap', max_mip_level=None):
    """PyTorch3D-based texture sampling implementation to replace nvdiffrast.
    
    Args:
        tex (torch.Tensor): Texture tensor [B, H, W, C] or [B, 6, H, W, C] for cubemaps
        uv (torch.Tensor): UV coordinates [B, H, W, 2] or [B, H, W, 3] for cubemaps
        uv_da (torch.Tensor, optional): UV derivatives for mipmapping
        mip_level_bias (torch.Tensor, optional): Per-pixel mip level bias
        mip (Union[List[torch.Tensor], None], optional): Custom mipmap stack
        filter_mode (str): 'auto', 'nearest', 'linear', 'linear-mipmap-nearest', 'linear-mipmap-linear'
        boundary_mode (str): 'wrap', 'clamp', 'zero', 'cube'
        max_mip_level (int, optional): Maximum mip level to use
        
    Returns:
        torch.Tensor: Sampled texture values [B, H, W, C]
    """
    if filter_mode == 'auto':
        filter_mode = 'linear-mipmap-linear' if (uv_da is not None or mip_level_bias is not None) else 'linear'
    
    # Handle boundary modes
    if boundary_mode == 'wrap':
        uv = uv - torch.floor(uv)
    elif boundary_mode == 'clamp':
        uv = torch.clamp(uv, 0, 1)
    elif boundary_mode == 'zero':
        # Will be handled in sampling
        pass
    elif boundary_mode == 'cube':
        raise NotImplementedError("Cubemap texturing not yet implemented")
    
    # Convert UV coordinates to pixel coordinates
    h, w = tex.shape[1:3]
    uv_px = uv * torch.tensor([w-1, h-1], device=uv.device)
    
    if filter_mode == 'nearest':
        uv_px = torch.round(uv_px)
        return sample_nearest(tex, uv_px, boundary_mode)
        
    elif filter_mode == 'linear':
        return sample_bilinear(tex, uv_px, boundary_mode)
        
    elif filter_mode in ['linear-mipmap-nearest', 'linear-mipmap-linear']:
        if mip is None:
            mip = generate_mipmaps(tex, max_mip_level)
            
        if mip_level_bias is None and uv_da is not None:
            mip_level_bias = compute_mipmap_level(uv_da, tex.shape[1:3])
            
        if filter_mode == 'linear-mipmap-nearest':
            mip_level = torch.round(mip_level_bias)
            return sample_mipmap_nearest(tex, mip, uv_px, mip_level, boundary_mode)
        else:
            return sample_mipmap_linear(tex, mip, uv_px, mip_level_bias, boundary_mode)

def sample_nearest(tex, uv_px, boundary_mode):
    """Sample texture using nearest neighbor interpolation."""
    uv_px = torch.round(uv_px)
    
    if boundary_mode == 'zero':
        mask = ((uv_px[..., 0] >= 0) & (uv_px[..., 0] < tex.shape[2]) &
                (uv_px[..., 1] >= 0) & (uv_px[..., 1] < tex.shape[1]))
        uv_px = torch.clamp(uv_px, 0, torch.tensor([tex.shape[2]-1, tex.shape[1]-1],
                                                   device=uv_px.device))
        
    x, y = uv_px[..., 0].long(), uv_px[..., 1].long()
    sampled = tex[:, y, x]
    
    if boundary_mode == 'zero':
        sampled = sampled * mask.unsqueeze(-1).float()
        
    return sampled

def sample_bilinear(tex, uv_px, boundary_mode):
    """Sample texture using bilinear interpolation."""
    # Get corner pixels
    uv0 = torch.floor(uv_px)
    uv1 = uv0 + 1
    
    # Calculate weights
    w1 = uv_px - uv0
    w0 = 1 - w1
    
    if boundary_mode == 'zero':
        mask = ((uv_px[..., 0] >= 0) & (uv_px[..., 0] < tex.shape[2]-1) &
                (uv_px[..., 1] >= 0) & (uv_px[..., 1] < tex.shape[1]-1))
        uv0 = torch.clamp(uv0, 0, torch.tensor([tex.shape[2]-1, tex.shape[1]-1],
                                              device=uv_px.device))
        uv1 = torch.clamp(uv1, 0, torch.tensor([tex.shape[2]-1, tex.shape[1]-1],
                                              device=uv_px.device))
    
    # Sample four corners
    x0, y0 = uv0[..., 0].long(), uv0[..., 1].long()
    x1, y1 = uv1[..., 0].long(), uv1[..., 1].long()
    
    c00 = tex[:, y0, x0]
    c10 = tex[:, y0, x1]
    c01 = tex[:, y1, x0]
    c11 = tex[:, y1, x1]
    
    # Interpolate
    c0 = c00 * w0[..., 0:1] + c10 * w1[..., 0:1]
    c1 = c01 * w0[..., 0:1] + c11 * w1[..., 0:1]
    sampled = c0 * w0[..., 1:2] + c1 * w1[..., 1:2]
    
    if boundary_mode == 'zero':
        sampled = sampled * mask.unsqueeze(-1).float()
        
    return sampled

def generate_mipmaps(tex, max_level=None):
    """Generate mipmap pyramid for a texture."""
    mips = [tex]
    h, w = tex.shape[1:3]
    
    current_h, current_w = h, w
    level = 0
    
    while current_h > 1 and current_w > 1 and (max_level is None or level < max_level):
        current_tex = mips[-1]
        next_h, next_w = current_h // 2, current_w // 2
        
        # Average 2x2 blocks
        next_mip = F.avg_pool2d(current_tex.permute(0, 3, 1, 2), 
                               kernel_size=2, stride=2)
        next_mip = next_mip.permute(0, 2, 3, 1)
        
        mips.append(next_mip)
        current_h, current_w = next_h, next_w
        level += 1
        
    return mips

def compute_mipmap_level(uv_da, tex_size):
    """Compute appropriate mipmap level from UV derivatives."""
    # Convert UV derivatives to pixel space
    duv_dx = uv_da[..., :2] * tex_size[1]
    duv_dy = uv_da[..., 2:] * tex_size[0]
    
    # Compute level based on maximum rate of change
    max_deriv = torch.maximum(
        torch.sum(duv_dx * duv_dx, dim=-1),
        torch.sum(duv_dy * duv_dy, dim=-1)
    )
    
    return 0.5 * torch.log2(max_deriv)

def sample_mipmap_nearest(tex, mip_stack, uv_px, mip_level, boundary_mode):
    """Sample from nearest mipmap level using bilinear filtering."""
    level = torch.clamp(mip_level, 0, len(mip_stack)-1).long()
    scaled_uv = uv_px / (2 ** level.unsqueeze(-1))
    
    results = []
    for i in range(level.max() + 1):
        mask = level == i
        if not mask.any():
            continue
            
        current_tex = mip_stack[i]
        current_uv = scaled_uv[mask]
        
        sample = sample_bilinear(current_tex, current_uv, boundary_mode)
        results.append((mask, sample))
        
    # Combine results
    output = torch.zeros_like(tex[:, :uv_px.shape[1], :uv_px.shape[2]])
    for mask, sample in results:
        output[mask] = sample
        
    return output

def sample_mipmap_linear(tex, mip_stack, uv_px, mip_level, boundary_mode):
    """Optimized mipmap sampling using grid_sample."""
    level_0 = torch.floor(mip_level)
    level_1 = torch.ceil(mip_level)
    frac = (mip_level - level_0).unsqueeze(-1)
    
    level_0 = torch.clamp(level_0, 0, len(mip_stack)-1).long()
    level_1 = torch.clamp(level_1, 0, len(mip_stack)-1).long()
    
    # Handle padding modes
    if boundary_mode == 'zero':
        padding_mode = 'zeros'
    elif boundary_mode == 'clamp':
        padding_mode = 'border'
    else:  # wrap mode
        padding_mode = 'border'
    
    # Convert UV coordinates to grid coordinates [-1, 1]
    grid = uv_px.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / (tex.shape[2] - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (tex.shape[1] - 1) - 1.0
    
    if boundary_mode == 'wrap':
        grid = ((grid + 1.0) % 2.0) - 1.0
    
    # Keep original batch shape for output
    batch_size, h, w, _ = grid.shape
    output_0 = torch.zeros(batch_size, h, w, tex.shape[-1], device=tex.device)
    output_1 = torch.zeros_like(output_0)

    # Sample each unique level
    for lvl in level_0.unique():
        mask = level_0 == lvl
        if not mask.any():
            continue
            
        current_tex = mip_stack[lvl].permute(0, 3, 1, 2)  # [B, C, H, W]
        sample = F.grid_sample(
            current_tex,
            grid.reshape(batch_size, h, w, 2),  # [B, H, W, 2]
            mode='bilinear',
            padding_mode=padding_mode,
            align_corners=False
        )
        output_0[mask.reshape(batch_size, h, w)] = sample.permute(0, 2, 3, 1)[mask.reshape(batch_size, h, w)]

    for lvl in level_1.unique():
        mask = level_1 == lvl
        if not mask.any():
            continue
            
        current_tex = mip_stack[lvl].permute(0, 3, 1, 2)  # [B, C, H, W]
        sample = F.grid_sample(
            current_tex,
            grid.reshape(batch_size, h, w, 2),  # [B, H, W, 2]
            mode='bilinear',
            padding_mode=padding_mode,
            align_corners=False
        )
        output_1[mask.reshape(batch_size, h, w)] = sample.permute(0, 2, 3, 1)[mask.reshape(batch_size, h, w)]

    return output_0 * (1 - frac.reshape(batch_size, h, w, 1)) + output_1 * frac.reshape(batch_size, h, w, 1)