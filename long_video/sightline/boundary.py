"""Known-clean shared-boundary constraints for flow matching and sampling."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _resize(value: torch.Tensor, spatial: tuple[int, int], *, noise_scale: bool = False) -> torch.Tensor:
    if tuple(value.shape[-2:]) == tuple(spatial):
        return value
    old_h = value.shape[-2]
    batch, channels, frames = value.shape[:3]
    result = F.interpolate(value.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, *value.shape[-2:]).float(),
                           size=spatial, mode="bilinear", align_corners=False)
    result = result.reshape(batch, frames, channels, *spatial).permute(0, 2, 1, 3, 4).to(value.dtype)
    return result * (old_h / spatial[0] if noise_scale else 1.0)


def boundary_at_sigma(clean: torch.Tensor, noise: torch.Tensor, sigma, spatial: tuple[int, int]) -> torch.Tensor:
    clean = _resize(clean, spatial)
    noise = _resize(noise, spatial, noise_scale=True)
    sigma = torch.as_tensor(sigma, device=clean.device, dtype=clean.dtype)
    while sigma.ndim < clean.ndim:
        sigma = sigma.unsqueeze(-1)
    return sigma * noise + (1.0 - sigma) * clean


def constrain_flow_item(item: dict, clean_boundary: torch.Tensor) -> dict:
    """Give training temporal0 the exact same known-boundary semantics as sampling."""
    noisy = item["noisy_latents"].clone()
    target = item["target"].clone()
    noise = item.get("noise")
    if noise is None:
        sigma = item["sigmas"]
        clean = item.get("clean_latents")
        if clean is None:
            raise ValueError("flow item must expose noise or clean_latents")
        noise = (noisy - (1.0 - sigma) * clean) / sigma.clamp_min(1e-8)
    clean = _resize(clean_boundary, tuple(noisy.shape[-2:]))
    boundary_noise = _resize(noise[:, :, :1], tuple(noisy.shape[-2:]), noise_scale=True)
    noisy[:, :, :1] = boundary_at_sigma(clean_boundary, noise[:, :, :1], item["sigmas"], tuple(noisy.shape[-2:]))
    target[:, :, :1] = boundary_noise - clean
    return {**item, "noisy_latents": noisy, "target": target, "known_clean_boundary": clean}


def _scheduler_sigma(scheduler, timestep, *, after_step: bool) -> torch.Tensor:
    times = scheduler.timesteps
    needle = torch.as_tensor(timestep, device=times.device).reshape(-1)[0]
    matches = torch.nonzero(torch.isclose(times.float(), needle.float()), as_tuple=False).flatten()
    if not len(matches):
        raise RuntimeError(f"scheduler timestep {float(needle)} has no sigma")
    index = int(matches[0]) + int(after_step)
    return scheduler.sigmas[min(index, len(scheduler.sigmas) - 1)]


def stage2_sample_with_boundary(pipe, *, clean_boundary: torch.Tensor | None, **kwargs):
    """Clamp temporal0 before every Transformer call and after every scheduler step."""
    if clean_boundary is None:
        return pipe.stage2_sample(**kwargs)
    initial = kwargs["latents"]
    boundary_noise = initial[:, :, :1].detach().clone()
    user_callback = kwargs.pop("callback_on_step_end", None)

    def pre_hook(_module, args, call_kwargs):
        hidden = call_kwargs.get("hidden_states")
        if hidden is None:
            return args, call_kwargs
        sigma = _scheduler_sigma(pipe.scheduler, call_kwargs["timestep"], after_step=False)
        hidden = hidden.clone()
        hidden[:, :, :1] = boundary_at_sigma(clean_boundary, boundary_noise, sigma, tuple(hidden.shape[-2:]))
        call_kwargs = dict(call_kwargs); call_kwargs["hidden_states"] = hidden
        return args, call_kwargs

    def callback(owner, step, timestep, callback_kwargs):
        if user_callback is not None:
            callback_kwargs = user_callback(owner, step, timestep, callback_kwargs)
        latents = callback_kwargs["latents"].clone()
        sigma = _scheduler_sigma(pipe.scheduler, timestep, after_step=True)
        latents[:, :, :1] = boundary_at_sigma(clean_boundary, boundary_noise, sigma, tuple(latents.shape[-2:]))
        return {**callback_kwargs, "latents": latents}

    handle = pipe.transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        return pipe.stage2_sample(callback_on_step_end=callback, **kwargs)
    finally:
        handle.remove()
