import numpy as np

DEFAULT_SOURCE_PRIOR = {0: 1.0, 1: 0.4, 2: 0.25, 3: 0.9, 4: 0.0}

def point_confidence(source, image_confidence, depth_confidence, view_angle_confidence=1.0, reprojection_confidence=1.0, priors=None):
    priors = priors or DEFAULT_SOURCE_PRIOR
    prior = np.asarray([priors.get(int(s), 0.0) for s in np.asarray(source).ravel()], dtype=np.float32).reshape(np.asarray(source).shape)
    return np.clip(prior * image_confidence * depth_confidence * view_angle_confidence * reprojection_confidence, 0, 1).astype(np.float32)
