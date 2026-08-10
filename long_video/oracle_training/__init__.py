"""Oracle-initialized single-scene WAH adaptation components."""

from .oracle_node import build_oracle_erp_node
from .temporal import ChunkContract, build_primary_loss_masks

from .revisit import (
    MultiChunkContract, scan_holo360d_zip,
    select_large_motion_windows, select_revisit_windows,
)
from .history_bank import HistoryBankKey, history_bank_cache_key, validate_history_bank_entry
from .round_robin import RoundRobinChunkScheduler, eligible_current_chunks
from .spatial_memory_prefix import SpatialMemoryBank, SpatialMemoryPrefixBank, choose_prefix
from .spatial_memory_warp import SpatialMemoryWarpBank
from .supervision import validate_current_chunk_supervision
from .causal_warp import CausalActiveNodeRenderer, CausalWarpResult

__all__ = [
    "build_oracle_erp_node", "ChunkContract", "build_primary_loss_masks",
    "MultiChunkContract", "scan_holo360d_zip",
    "select_large_motion_windows", "select_revisit_windows",
    "HistoryBankKey", "history_bank_cache_key", "validate_history_bank_entry",
    "RoundRobinChunkScheduler", "eligible_current_chunks",
    "SpatialMemoryPrefixBank", "choose_prefix",
    "SpatialMemoryWarpBank",
    "SpatialMemoryBank",
    "validate_current_chunk_supervision",
    "CausalActiveNodeRenderer", "CausalWarpResult",
]
