"""Online causal-world components.

Submodules are intentionally not imported here: ``MemoryManager`` depends on
``online.transition_buffer``, so eager renderer exports would create a cycle.
"""
