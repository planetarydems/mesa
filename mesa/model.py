"""The MESA network.

A U-Net (Ronneberger et al., 2015) that maps a CTX image, the MOLA gridded DEM
and the shot-distance layer to elevation and albedo. Two additions to the plain
U-Net, both described in Section 2.3 of the paper:

1. Illumination conditioning. The solar vector is expanded into sinusoids of
   increasing frequency and turned into a per-channel scale and shift (FiLM,
   Perez et al., 2018) for each encoder stage. The MOLA relief of the window,
   rho = (log10 sigma_MOLA - 2) / 1.5, enters the same pathway.

2. Fusion with MOLA. The network output u is combined with the standardised
   MOLA DEM m by solving

       z = argmin_z ||z - u||^2 + sum_p s_p (z_p - m_p)^2 + mu ||grad z||^2

   (Eq. 3 of the paper). The trust s_p >= 0 is learned from the shot-distance
   layer and from relief features, and mu >= 0 is a learned scalar. The
   minimiser solves a symmetric positive-definite linear system, which is
   solved with a fixed number of conjugate-gradient steps so that gradients
   flow through the solve.

The decoder doubles the resolution four times. The model of the paper (epoch
98, `mesa_e098.pt`) upsamples with a fixed degree-4 B-spline followed by a
learned 3 x 3 convolution (`upsample_mode = spline_conv`). The model of release
1 (epoch 88, `mesa_e088.pt`) used 2 x 2 transposed convolutions of stride 2
(`deconv`), which write every coarse pixel into its own 2 x 2 block and leave a
16/8/4/2 px lattice in the output. Both encoders downsample by 2 x 2 max-pooling
(`down_mode = maxpool`). The architecture is read from a configuration
(mesa/configs/*.json), never from the weights: the spline decoder has the
same parameter shapes as a bilinear one, so the weights alone cannot tell
which interpolator they were trained with.

Tensor layout throughout: [batch, channel, row, column], rows increasing to the
south and columns to the east. Elevations are standardised per window.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Relief scalar: rho = (log10(sigma_MOLA) - RELIEF_CENTER) / RELIEF_SCALE.
RELIEF_CENTER = 2.0
RELIEF_SCALE = 1.5
PIXEL_SIZE_M = 18.0


def relief_rho(mola_std_m):
    """Relief scalar of a window from the standard deviation of MOLA (metres)."""
    return (math.log10(max(float(mola_std_m), 1e-3)) - RELIEF_CENTER) / RELIEF_SCALE


def gaussian_blur(x, sigma):
    """Separable Gaussian blur of a [B, 1, H, W] tensor, edges replicated."""
    radius = max(1, int(round(3.0 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode='replicate'), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode='replicate'), k.view(1, 1, -1, 1))
    return x


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------

def encode_relief(rho):
    """[B, 1] -> [B, 7]: rho and three low-frequency sinusoid pairs. Low
    frequencies only, because rho varies smoothly from window to window."""
    out = [rho]
    for k in (1.0, 2.0, 4.0):
        out += [torch.sin(k * math.pi * rho), torch.cos(k * math.pi * rho)]
    return torch.cat(out, dim=1)


class FiLMGenerator(nn.Module):
    """Solar vector [B, 3] -> (gamma, beta) for each encoder stage.

    Each component of the vector is encoded as sin and cos of 2^k * pi * x for
    k = 0..15 (Mildenhall et al., 2020), giving 96 numbers, followed by a
    two-layer MLP and one linear head per stage. The heads start at gamma = 1,
    beta = 0, so the modulation is the identity at initialisation.
    """

    def __init__(self, condition_dim=3, hidden_dim=128, feature_dims=(64, 128, 256, 512),
                 n_frequencies=16):
        super().__init__()
        self.condition_dim = condition_dim
        self.n_frequencies = n_frequencies
        self.trunk = nn.Sequential(
            nn.Linear(condition_dim * n_frequencies * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.heads = nn.ModuleDict()
        for i, d in enumerate(feature_dims):
            head = nn.Linear(hidden_dim, 2 * d)
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias[:d], 1.0)
            nn.init.zeros_(head.bias[d:])
            self.heads[f'layer_{i}'] = head

    def encode(self, v):
        freqs = (2.0 ** torch.arange(self.n_frequencies, device=v.device, dtype=v.dtype)) * math.pi
        angles = v.unsqueeze(-1) * freqs.view(1, 1, -1)                 # [B, 3, F]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(1)

    def forward(self, sun, relief_embedding):
        # The relief embedding is added to the first layer's pre-activation, so the
        # first layer sees [encoded sun, encoded relief].
        h = self.trunk[0](self.encode(sun)) + relief_embedding
        for layer in list(self.trunk)[1:]:
            h = layer(h)
        params = {}
        for name, head in self.heads.items():
            out = head(h)
            d = out.shape[1] // 2
            params[name] = (out[:, :d, None, None], out[:, d:, None, None])
        return params


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two 3x3 convolutions with GroupNorm and GELU, a residual connection, and
    optional FiLM modulation after the first normalisation."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, film=None):
        h = self.norm1(self.conv1(x))
        if film is not None:
            gamma, beta = film
            h = gamma * h + beta
        h = self.dropout(self.act(h))
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.proj(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel weighting at the bottleneck."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.GELU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.avg_pool(x).flatten(1))
        return x * w[:, :, None, None]


class SplineUpsample2d(nn.Module):
    """Fixed x2 upsampling by a degree-4 B-spline: zero insertion followed by the
    separable kernel [1, 5, 10, 10, 5, 1] / 16 (= [1, 1] convolved five times),
    with replicated borders. Each output phase sums to one, so constants are
    exact and linear fields are reproduced; outputs sit where bilinear
    interpolation with align_corners=False puts them. No parameters (the kernel
    is a non-persistent buffer)."""

    def __init__(self):
        super().__init__()
        k = torch.tensor([1.0, 5.0, 10.0, 10.0, 5.0, 1.0]) / 16.0
        self.register_buffer('kernel', (k[:, None] * k[None, :]).view(1, 1, 6, 6), persistent=False)

    def forward(self, x):
        c = x.shape[1]
        xp = F.pad(x, (1, 1, 1, 1), mode='replicate')
        return F.conv_transpose2d(xp, self.kernel.to(x.dtype).expand(c, 1, 6, 6), stride=2, padding=4, groups=c)


def upsampler(mode, in_ch, out_ch):
    """One x2 step of the decoder."""
    if mode == 'deconv':
        return nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
    if mode == 'spline_conv':
        return nn.Sequential(SplineUpsample2d(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
    raise ValueError(f'unknown upsample_mode {mode!r} (deconv or spline_conv)')


def downsampler(mode):
    """One /2 step of the encoder."""
    if mode == 'maxpool':
        return nn.MaxPool2d(2)
    raise ValueError(f'unknown down_mode {mode!r} (maxpool)')


# ---------------------------------------------------------------------------
# Fusion with MOLA (Eq. 3)
# ---------------------------------------------------------------------------

def laplacian(z):
    """5-point Laplacian with Neumann (replicated) edges. Constants map to zero."""
    zp = F.pad(z, (1, 1, 1, 1), mode='replicate')
    return (4.0 * z - zp[:, :, 1:-1, :-2] - zp[:, :, 1:-1, 2:]
            - zp[:, :, :-2, 1:-1] - zp[:, :, 2:, 1:-1])


def conjugate_gradient(b, s, mu, n_iter, eps=1e-12):
    """Solve (I + diag(s) + mu L) z = b from z = 0, independently per sample.

    The loop is unrolled, so autograd differentiates through it."""
    def A(v):
        return v + s * v + mu * laplacian(v)

    def dot(a, c):
        return (a * c).sum(dim=(1, 2, 3), keepdim=True)

    z = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = dot(r, r)
    for _ in range(n_iter):
        Ap = A(p)
        alpha = rs / dot(p, Ap).clamp_min(eps)
        z = z + alpha * p
        r = r - alpha * Ap
        rs_new = dot(r, r)
        p = r + (rs_new / rs.clamp_min(eps)) * p
        rs = rs_new
    return z


def relief_features(m, conf, rho):
    """Per-pixel relief features for the MOLA trust, in physical units.

    m is standardised MOLA, conf the shot-distance layer, rho the window's
    relief scalar. MOLA slope in metres per metre is |grad m| sigma_MOLA / 18 m,
    smoothed with a Gaussian of 4 pixels so the facets of the bilinear MOLA
    surface do not dominate, and mapped to rho_loc = (log10(tan slope) + 1.5)/1.5.
    Returns [rho, rho_loc, conf rho, conf rho_loc, rho rho_loc].
    """
    rho = rho.view(m.shape[0], 1, 1, 1).to(m.dtype)
    std_m = torch.pow(10.0, rho * RELIEF_SCALE + RELIEF_CENTER)
    mp = F.pad(m, (1, 1, 1, 1), mode='replicate')
    gx = 0.5 * (mp[:, :, 1:-1, 2:] - mp[:, :, 1:-1, :-2])
    gy = 0.5 * (mp[:, :, 2:, 1:-1] - mp[:, :, :-2, 1:-1])
    tan_slope = gaussian_blur(torch.sqrt(gx * gx + gy * gy + 1e-12) * std_m / PIXEL_SIZE_M, 4.0)
    rho_loc = ((torch.log10(tan_slope + 3e-4) + 1.5) / 1.5).clamp(-1.5, 1.5)
    rho_tile = rho.expand_as(m)
    return torch.cat([rho_tile, rho_loc, conf * rho_tile, conf * rho_loc, rho_tile * rho_loc], dim=1)


def softplus_inverse(y):
    return math.log(math.expm1(y))


class MolaFusion(nn.Module):
    """Solves Eq. 3 for the fused standardised elevation z.

    s = softplus(conv1x1(conf) + conv1x1(relief features)) and mu = softplus(mu_raw).
    At initialisation s = 1 everywhere and mu = 0.05. The solve always runs in
    float32 with autocast disabled.
    """

    def __init__(self, n_iter=25, mu_init=0.05, s_init=1.0):
        super().__init__()
        self.cg_iters = int(n_iter)
        self.s_conv = nn.Conv2d(1, 1, 1)
        nn.init.zeros_(self.s_conv.weight)
        nn.init.constant_(self.s_conv.bias, softplus_inverse(s_init))
        self.rel_conv = nn.Conv2d(5, 1, 1, bias=False)
        nn.init.zeros_(self.rel_conv.weight)
        self.mu_raw = nn.Parameter(torch.tensor(softplus_inverse(mu_init)))

    def trust(self, conf, rel):
        return F.softplus(self.s_conv(conf) + self.rel_conv(rel))

    def forward(self, u, m, conf, rel):
        dt = self.mu_raw.dtype
        with torch.autocast(device_type=u.device.type, enabled=False):
            s = self.trust(conf.to(dt), rel.to(dt))
            mu = F.softplus(self.mu_raw)
            z = conjugate_gradient(u.to(dt) + s * m.to(dt), s, mu, self.cg_iters)
        return z.to(u.dtype)


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

class MESA(nn.Module):
    """U-Net with illumination conditioning, a DEM + albedo head and MOLA fusion.

    forward(ctx, x, cond) with
        ctx   [B, 1, H, W]  image stretched to [0, 1]
        x     [B, 2, H, W]  standardised MOLA, then the shot-distance layer
                            d_shot = exp(-d1 / 2000 m)
        cond  [B, 4]        solar vector (east, north, up) followed by rho
    returns (z, albedo): standardised elevation and albedo in [0, 1].
    H and W should be multiples of 16. upsample_mode and down_mode select the
    resampling of the decoder and the encoder (module docstring); the defaults
    are those of epoch 88.
    """

    def __init__(self, in_channels=3, depths=(64, 128, 256, 512), dropout=0.1, cg_iters=25,
                 upsample_mode='deconv', down_mode='maxpool'):
        super().__init__()
        depths = list(depths)
        self.depths = depths
        self.config = dict(in_channels=int(in_channels), depths=depths, dropout=float(dropout),
                           cg_iters=int(cg_iters), upsample_mode=upsample_mode, down_mode=down_mode)
        self.film_gen = FiLMGenerator(condition_dim=3, hidden_dim=128, feature_dims=depths)

        self.enc_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        prev = in_channels
        for d in depths:
            self.enc_blocks.append(ConvBlock(prev, d, dropout))
            self.downsample.append(downsampler(down_mode))
            prev = d
        self.bottleneck = ConvBlock(depths[-1], depths[-1], dropout)
        self.bottleneck_attn = ChannelAttention(depths[-1])

        self.upsample = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i, d in enumerate(reversed(depths)):
            up_in = depths[-1] if i == 0 else depths[len(depths) - i]
            self.upsample.append(upsampler(upsample_mode, up_in, d))
            self.dec_blocks.append(ConvBlock(2 * d, d, dropout))

        # channel 0: elevation estimate u; channel 1: albedo logit
        self.output = nn.Sequential(
            nn.Conv2d(depths[0], depths[0] // 2, 3, padding=1),
            nn.GroupNorm(min(8, depths[0] // 2), depths[0] // 2),
            nn.GELU(),
            nn.Conv2d(depths[0] // 2, 2, 1),
        )
        self.implicit_solve = MolaFusion(n_iter=cg_iters)
        # relief scalar -> first FiLM layer (zero at initialisation)
        self.scale_proj = nn.Linear(7, 128, bias=False)
        nn.init.zeros_(self.scale_proj.weight)

    def forward(self, ctx, x, cond):
        sun, rho = cond[:, :3], cond[:, 3:4]
        film = self.film_gen(sun, self.scale_proj(encode_relief(rho)))

        h = torch.cat([ctx, x], dim=1)
        skips = []
        for i, (enc, down) in enumerate(zip(self.enc_blocks, self.downsample)):
            h = enc(h, film[f'layer_{i}'])
            skips.append(h)
            h = down(h)
        h = self.bottleneck_attn(self.bottleneck(h))
        for up, dec, skip in zip(self.upsample, self.dec_blocks, reversed(skips)):
            h = up(h)
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            h = dec(torch.cat([h, skip], dim=1))

        out = self.output(h)
        m, conf = x[:, :1], x[:, -1:]
        rel = relief_features(m.float(), conf.float(), rho.float())
        z = self.implicit_solve(out[:, 0:1], m, conf, rel)
        return z, torch.sigmoid(out[:, 1:2])


CONFIG_DIR = Path(__file__).resolve().parent / 'configs'
ARCH_KEYS = ('in_channels', 'depths', 'dropout', 'cg_iters', 'upsample_mode', 'down_mode')
# Released weight files and their configurations. Release v1.0.0 published the
# epoch-88 weights as `mader_e088.pt`; the file is the same as `mesa_e088.pt`.
RELEASED = {'mesa_e098.pt': 'mesa_e098.json', 'mesa_e088.pt': 'mesa_e088.json', 'mader_e088.pt': 'mesa_e088.json'}


def read_config(config):
    """A model configuration from a dict, a JSON file or the name of a packaged
    configuration (`mesa_e098`, `mesa_e088`)."""
    if isinstance(config, dict):
        cfg = dict(config)
    else:
        p = Path(config)
        if not p.exists() and (CONFIG_DIR / f'{p.stem}.json').exists():
            p = CONFIG_DIR / f'{p.stem}.json'
        cfg = json.loads(p.read_text())
    missing = [k for k in ('upsample_mode', 'down_mode') if k not in cfg]
    if missing:
        raise ValueError(f'model configuration without {missing}')
    return cfg


def build(config):
    """MESA built from a configuration (read_config)."""
    cfg = read_config(config)
    return MESA(**{k: cfg[k] for k in ARCH_KEYS if k in cfg})


def weights_path(path):
    """The weight file to load. A missing `mesa_e088.pt` falls back to
    `mader_e088.pt` in the same directory, the name of the v1.0.0 release
    asset, and the other way round."""
    p = Path(path)
    if p.exists() or p.name not in ('mesa_e088.pt', 'mader_e088.pt'):
        return p
    for name in ('mesa_e088.pt', 'mader_e088.pt'):
        if p.with_name(name).exists():
            return p.with_name(name)
    return p


def strip_prefixes(state):
    return {k.removeprefix('_forward_module.').removeprefix('module.'): v for k, v in state.items()}


def resolve_config(path, ckpt, config=None):
    """The configuration of a weight file, in this order: the `config` argument;
    the configuration stored in the file (the released weights and the files
    written by train.py carry one); `<weights>.json` beside the file; the
    packaged configuration of a released file name. A state dict with
    transposed convolutions (`upsample.0.weight`) and no configuration is the
    epoch-88 network. Any other decoder without a configuration is refused,
    because its interpolator cannot be read from the weights."""
    if config is not None:
        return read_config(config)
    if isinstance(ckpt, dict) and isinstance(ckpt.get('config'), dict):
        return read_config(ckpt['config'])
    p = Path(path)
    if p.with_suffix('.json').exists():
        return read_config(p.with_suffix('.json'))
    if p.name in RELEASED:
        return read_config(CONFIG_DIR / RELEASED[p.name])
    state = strip_prefixes(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    if 'upsample.0.weight' in state:
        return read_config(CONFIG_DIR / 'mesa_e088.json')
    raise ValueError(f'{path}: no model configuration; pass config= (for example mesa/configs/mesa_e098.json)')


def load_model(path, device='cpu', config=None):
    """Build MESA from its configuration (resolve_config) and load the weights:
    a released file (`mesa_e098.pt`, the model of the paper, or `mesa_e088.pt`,
    also accepted under its v1.0.0 name `mader_e088.pt`) or a file written by
    train.py. Refuses a partial load."""
    p = weights_path(path)
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    cfg = resolve_config(p, ckpt, config)
    state = strip_prefixes(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    model = build(cfg)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
