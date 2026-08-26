"""Compatibility entry point for the genuine-2D TFIM example.

The former cylinder/VUMPS comparison has been removed because it compared
one-dimensional column-blocked ansatzes rather than states on the infinite
two-dimensional plane.
"""

from .tfim_plane import PlaneRun, main, run_tfim_plane

__all__ = ["PlaneRun", "main", "run_tfim_plane"]


if __name__ == "__main__":
    main()
