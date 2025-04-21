import torch
import einops
from modules import devices


latent_rgb_factors = [ # from comfyui
    [-0.0395, -0.0331, 0.0445],
    [0.0696, 0.0795, 0.0518],
    [0.0135, -0.0945, -0.0282],
    [0.0108, -0.0250, -0.0765],
    [-0.0209, 0.0032, 0.0224],
    [-0.0804, -0.0254, -0.0639],
    [-0.0991, 0.0271, -0.0669],
    [-0.0646, -0.0422, -0.0400],
    [-0.0696, -0.0595, -0.0894],
    [-0.0799, -0.0208, -0.0375],
    [0.1166, 0.1627, 0.0962],
    [0.1165, 0.0432, 0.0407],
    [-0.2315, -0.1920, -0.1355],
    [-0.0270, 0.0401, -0.0821],
    [-0.0616, -0.0997, -0.0727],
    [0.0249, -0.0469, -0.1703]
]
latent_rgb_factors_bias = [0.0259, -0.0192, -0.0761]
vae_weight = None
vae_bias = None


def vae_decode_simple(latents):
    global vae_weight, vae_bias # pylint: disable=global-statement
    with devices.inference_context():
        if vae_weight is None or vae_bias is None:
            vae_weight = torch.tensor(latent_rgb_factors, device=devices.device, dtype=devices.dtype).transpose(0, 1)[:, :, None, None, None]
            vae_bias = torch.tensor(latent_rgb_factors_bias, device=devices.device, dtype=devices.dtype)
        images = torch.nn.functional.conv3d(latents.to(devices.dtype), weight=vae_weight, bias=vae_bias, stride=1, padding=0, dilation=1, groups=1)
        images = (images + 1.1) * 110 # sort-of normalized
        images = einops.rearrange(images, 'b c t h w -> (b h) (t w) c')
        images = images.to(torch.uint8).detach().cpu().numpy().clip(0, 255)
    return images
