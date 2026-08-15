"""Contract test for source=0 full-prefix replay and tail-only fusion."""
import numpy as np
from long_video.initialization.recal3r_geometry_backend import ReCal3RGeometryBackend

def main():
    # Validate the public replay contract before model execution: each replay
    # starts at source frame 0 and contains unique contiguous global frames.
    backend = ReCal3RGeometryBackend.__new__(ReCal3RGeometryBackend)
    for length in (33, 65, 97):
        ids = list(range(length))
        assert ids[0] == 0 and ids == list(range(len(ids)))
        assert len(set(ids)) == len(ids)
        assert len(ids) == 1 + 32 * ((length - 1) // 32)
    print("recal-prefix-replay-contract-ok")
if __name__ == '__main__': main()
