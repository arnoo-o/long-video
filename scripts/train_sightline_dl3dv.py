#!/usr/bin/env python3
"""Legacy import/CLI alias; formal training lives in train_sightline_rgbd.py."""
if __package__:
    from . import train_sightline_rgbd as _implementation
else:
    import train_sightline_rgbd as _implementation

globals().update({name: getattr(_implementation, name) for name in dir(_implementation) if not name.startswith("__")})


if __name__ == "__main__":
    main()
