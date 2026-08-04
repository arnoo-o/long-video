import numpy as np


def confidence_bias(token_confidence, lambda_conf=1.0, eps=1e-6):
    """Return additive key logits bias; confidence=1 exactly maps to zero."""
    try:
        import torch
        if torch.is_tensor(token_confidence):
            return float(lambda_conf) * torch.log(token_confidence.clamp(min=float(eps), max=1.0))
    except ImportError:
        pass
    return float(lambda_conf) * np.log(np.clip(token_confidence, eps, 1.0))


def attention_kwargs(lambda_confidence=1.0, confidence_threshold=0.0, epsilon=1e-6):
    return {
        "history_confidence_lambda": float(lambda_confidence),
        "history_confidence_threshold": float(confidence_threshold),
        "history_confidence_epsilon": float(epsilon),
    }
