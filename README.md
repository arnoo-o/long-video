# long-video

Causal long-video inference uses one compact path:

1. ReCal3R builds and grows the causal point-cloud world from the source and prior generated chunks.
2. Parent-First renderer produces aligned warp RGB, binary visibility, and confidence.
3. Pinned original Warp-as-History consumes the rendered warp as history conditioning.
4. Helios samples three pyramid stages with `[2, 2, 4]` scheduler updates.
5. Only Stage2 steps 0, 1, and 2 apply renderer RGB consistency; Stage2 step 3 remains native Helios.

The clamp is `I_mixed = M * I_warp + (1-M) * I_model`, where `M` is raw binary
renderer visibility. It decodes the clean/x0 prediction, composites pixels,
deterministically VAE-encodes the result, and returns it to the native next
scheduler coordinate. There is no final post-hoc clamp.

DL3DV download, selection, manifest, and preprocessing tools are retained under
`scripts/select_download_dl3dv_film.py` and `scripts/build_dl3dv_film_dataset.py`.

Pinned WAH commit: `09aa6461355b298bfced51007bd709a251d6033a`.
