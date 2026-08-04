import numpy as np
def confidence_bias(token_confidence, lambda_conf=1.0, eps=1e-6): return lambda_conf*np.log(np.clip(token_confidence,eps,1.0))
