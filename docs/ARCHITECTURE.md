# Architecture

The runtime has four components only:

- pinned original Warp-as-History conditioning;
- frozen ReCal3R causal geometry and `MemoryManager` world growth;
- a Stage0-only spatial FiLM;
- native Helios Stage0/1/2 generation.

The current committed node is rendered with Parent-First composition. Its RGB
is deterministically VAE encoded and spatially downsampled to the real Stage0
latent shape. Renderer visibility is grouped as `[0], [1..4], ... [29..32]`
and downsampled continuously. The 16-channel world latent and one-channel
visibility are passed to a per-position `16 -> 32 -> 32` FiLM immediately
before Stage0 patch embedding. Exact tensor-shape matching prevents Stage1/2
modulation.

Only generated past frames enter `TransitionBuffer` and ReCal3R. A supervised
current/future RGB or depth tensor cannot enter the world builder. Accepted
candidate points preserve the complete parent, remain frozen as a shadow, and
activate two chunks after creation. Parent-First rendering gives the latest
delta only hole-filling rights.
