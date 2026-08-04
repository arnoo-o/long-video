class WAHAdapter:
    """Thin integration boundary; concrete WAH/Helios objects are injected."""
    def __init__(self, wah_pipeline=None, lambda_confidence=1.0): self.wah_pipeline=wah_pipeline; self.lambda_confidence=lambda_confidence
    def build_warp_history(self, warp_batch):
        from .token_confidence import build_token_confidence
        return warp_batch, build_token_confidence(warp_batch.confidence, warp_batch.visibility, warp_batch.confidence.shape, warp_batch.confidence.shape)
