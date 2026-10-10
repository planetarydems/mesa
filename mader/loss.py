"""Training objective (Section 2.2 of the paper).

    L = L1(z, z_ref) + 0.5 L1(Sobel z, Sobel z_ref)
        + 0.3 L1(A * shading(z), I) + 0.01 TV(A)

z and z_ref are standardised elevations of the prediction and the stereo DEM,
A the predicted albedo and I the image. The photometric term converts z back to
metres with the stereo DEM's statistics and renders Lambertian shading under the
known solar vector; it has no reference albedo, so the total-variation term
keeps A smooth.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def lambertian_shading(dem_m, sun, pixel_size=18.0, eps=1e-6):
    """cos of the local incidence angle, clamped at 0 (self-shadow).

    dem_m [B, 1, H, W] in metres, rows to the south; sun [B, 3] = (east, north, up).
    """
    dz_drow = (dem_m[:, :, 2:, :] - dem_m[:, :, :-2, :]) * 0.5 / pixel_size
    dz_dcol = (dem_m[:, :, :, 2:] - dem_m[:, :, :, :-2]) * 0.5 / pixel_size
    dz_drow = F.pad(dz_drow, (0, 0, 1, 1), mode='replicate')
    dz_dcol = F.pad(dz_dcol, (1, 1, 0, 0), mode='replicate')
    dz_dx, dz_dy = dz_dcol, -dz_drow                     # east, north
    inv_norm = 1.0 / torch.sqrt(dz_dx * dz_dx + dz_dy * dz_dy + 1.0 + eps)
    nx, ny, nz = -dz_dx * inv_norm, -dz_dy * inv_norm, inv_norm
    s = sun.view(-1, 3, 1, 1)
    return torch.clamp(nx * s[:, 0:1] + ny * s[:, 1:2] + nz * s[:, 2:3], min=0.0)


class MaderLoss(nn.Module):

    def __init__(self, grad_weight=0.5, photo_weight=0.3, tv_weight=0.01, pixel_size=18.0):
        super().__init__()
        self.grad_weight = grad_weight
        self.photo_weight = photo_weight
        self.tv_weight = tv_weight
        self.pixel_size = pixel_size
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_x.t().contiguous().view(1, 1, 3, 3))

    def forward(self, z, albedo, z_ref, image, sun, dem_mean, dem_std):
        """z, albedo, z_ref, image: [B, 1, H, W]; sun [B, 3]; dem_mean, dem_std [B]."""
        l1 = F.l1_loss(z, z_ref)

        gx = lambda a: F.conv2d(a, self.sobel_x, padding=1)
        gy = lambda a: F.conv2d(a, self.sobel_y, padding=1)
        grad = 0.5 * (F.l1_loss(gx(z), gx(z_ref)) + F.l1_loss(gy(z), gy(z_ref)))

        dem_m = z * dem_std.view(-1, 1, 1, 1) + dem_mean.view(-1, 1, 1, 1)
        photo = F.l1_loss(albedo * lambertian_shading(dem_m, sun, self.pixel_size), image)

        tv = ((albedo[:, :, :, 1:] - albedo[:, :, :, :-1]).abs().mean()
              + (albedo[:, :, 1:, :] - albedo[:, :, :-1, :]).abs().mean())

        total = l1 + self.grad_weight * grad + self.photo_weight * photo + self.tv_weight * tv
        return total, {'l1': l1.item(), 'grad': grad.item(), 'photo': photo.item(), 'tv': tv.item()}
