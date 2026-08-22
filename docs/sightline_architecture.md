# Sightline mainline

Sightline is the formal training and inference path. Its only runtime inputs are
the source RGB/latent, intrinsics, initial c2w, subsequent c2w or WASD controls,
and generated latent/history state. Every Helios token grid computes its own
Plücker rays from token centers; no interpolated fixed ray map is used.

The Q/K-only binding is applied after native Q/K normalization and RoPE. Each
selected layer owns independent Q/K projections, gates, and RMSNorm parameters;
values, FFN, cross-attention, and backbone activations are untouched. A single
scalar `alpha` is shared by all selected layers, zero-initialized, and is the only global gate. Short history is bounded
to source + 16 long + 2 mid + 1 short latent slots. Completed chunks may be
captured as K/V-only memory, with configurable budget and oldest-first eviction;
memory tokens can never become queries.

Correspondence teachers are offline sparse rows derived from existing teacher
arrays. They are never imported by inference. The training objective combines
exact flow matching with a probe-layer correspondence cross entropy and optional
LoRA on the selected layers.

The former WAH, PointWorld, ReCal3R and Pi3X implementations are legacy/reference
assets only. They are intentionally absent from Sightline's import dependency
chain and must not be initialized by the new scripts.
