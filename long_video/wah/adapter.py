class WAHAdapter:
    """Adapter for the patched official WAH external-warp API."""
    def __init__(self, wah_pipeline=None, lambda_confidence=1.0, confidence_threshold=0.0,
                 confidence_epsilon=1e-6):
        self.wah_pipeline = wah_pipeline
        self.lambda_confidence = float(lambda_confidence)
        self.confidence_threshold = float(confidence_threshold)
        self.confidence_epsilon = float(confidence_epsilon)

    @staticmethod
    def warp_inputs(warp_batch):
        import numpy as np
        rgb = np.asarray(warp_batch.rgb, np.float32)
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f"WarpBatch.rgb must be [T,H,W,3], got {rgb.shape}")
        visibility = np.asarray(warp_batch.visibility, np.float32)
        confidence = np.asarray(warp_batch.confidence, np.float32)
        if visibility.shape != rgb.shape[:3] or confidence.shape != rgb.shape[:3]:
            raise ValueError("WarpBatch visibility/confidence must match RGB [T,H,W]")
        return {
            "warp_video": rgb,
            "warp_visibility_mask": visibility[None, None],
            "warp_confidence_mask": (confidence * visibility)[None, None],
        }

    def configure_state(self, state):
        from .attention_bias import attention_kwargs
        kwargs = dict(state.get("attention_kwargs") or {})
        kwargs.update(attention_kwargs(
            self.lambda_confidence, self.confidence_threshold, self.confidence_epsilon
        ))
        state["attention_kwargs"] = kwargs
        return state

    def generate_next_chunk(self, state, warp_batch, output_type=None):
        if self.wah_pipeline is None:
            raise RuntimeError("WAHAdapter requires a patched official WarpAsHistoryPipeline.")
        self.configure_state(state)
        return self.wah_pipeline.generate_next_chunk(
            state, output_type=output_type, **self.warp_inputs(warp_batch)
        )

    def build_warp_history(self, warp_batch):
        """Expose unmodified warp pixels plus official-pipeline input tensors."""
        return warp_batch, self.warp_inputs(warp_batch)
