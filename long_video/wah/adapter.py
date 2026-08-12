class WAHAdapter:
    """Adapter for the patched official WAH external-warp API."""
    def __init__(self, wah_pipeline=None, lambda_confidence=1.0, confidence_threshold=0.0,
                 confidence_epsilon=1e-6):
        self.wah_pipeline = wah_pipeline
        self.lambda_confidence = float(lambda_confidence)
        self.confidence_threshold = float(confidence_threshold)
        self.confidence_epsilon = float(confidence_epsilon)
    @classmethod
    def from_config(cls,wah_pipeline,confidence_config):
        return cls(
            wah_pipeline=wah_pipeline,
            lambda_confidence=confidence_config["lambda_confidence"],
            confidence_threshold=confidence_config["token_confidence_threshold"],
            confidence_epsilon=float(confidence_config.get("epsilon",1e-6)),
        )


    @staticmethod
    def warp_inputs(warp_batch, fill_frame=None):
        import numpy as np
        rgb = np.asarray(warp_batch.rgb, np.float32)
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f"WarpBatch.rgb must be [T,H,W,3], got {rgb.shape}")
        visibility = np.asarray(warp_batch.visibility, np.float32)
        confidence = np.asarray(warp_batch.confidence, np.float32)
        if visibility.shape != rgb.shape[:3] or confidence.shape != rgb.shape[:3]:
            raise ValueError("WarpBatch visibility/confidence must match RGB [T,H,W]")
        if fill_frame is not None:
            source = np.asarray(fill_frame, np.float32)
            if source.ndim != 3 or source.shape[-1] != 3:
                raise ValueError(f"WAH fill frame must be [H,W,3], got {source.shape}")
            if source.max(initial=0.0) > 1.0:
                source = source / 255.0
            mean_rgb = source.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
            rgb = np.where(visibility[..., None] > 0, rgb, mean_rgb)
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

    def generate_next_chunk(self, state, warp_batch, output_type=None, fill_frame=None):
        if self.wah_pipeline is None:
            raise RuntimeError("WAHAdapter requires a patched official WarpAsHistoryPipeline.")
        self.configure_state(state)
        return self.wah_pipeline.generate_next_chunk(
            state, output_type=output_type, **self.warp_inputs(warp_batch, fill_frame=fill_frame)
        )

    def build_warp_history(self, warp_batch):
        """Expose unmodified warp pixels plus official-pipeline input tensors."""
        return warp_batch, self.warp_inputs(warp_batch)
