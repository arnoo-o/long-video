"""Validated configuration loader for the Sightline mainline."""
from dataclasses import dataclass
from pathlib import Path
import yaml
@dataclass(frozen=True)
class SightlineConfig:
    ray_epsilon: float; scale_augmentation_probability: float; scale_augmentation_range: tuple[float,float]; sightline_enabled:bool; alpha_init:float
    history_sizes: tuple[int,int,int]; chunk_length:int; chunk_stride:int; sightline_layers:tuple[int,...]; camera_layers:tuple[int,...]; memory_layers:tuple[int,...]; correspondence_layers:tuple[int,...]
    lora_layers:tuple[int,...]; lora_rank:int; memory_pool:int; memory_budget:int; lambda_corr:float; lambda_corr_final:float
    lambda_corr_decay_start:float; learning_rate:float; lora_learning_rate:float; warmup_ratio:float; grad_clip:float; bf16:bool; accumulation_steps:int; high_noise_bias:float; teacher_forcing_ratio:float; self_rollout_ratio:float; memory_write_sigma:float; correspondence_rows_per_batch:int; gradient_checkpointing:bool; diagnostics_frequency:int; phase:str; model_id:str; source_height:int; source_width:int; chunk_count:int; pyramid_steps:tuple[int,...]; data_path:str; latent_cache_path:str; correspondence_cache_path:str; output_path:str
    sightline_training_semantics_version:str; sightline_correspondence_schema_version:str; sightline_checkpoint_schema_version:str; p1_steps:int; p2_steps:int; p3_steps:int; ddp_world_size:int; checkpoint_every:int
def load_sightline_config(path: str|Path) -> SightlineConfig:
    raw=yaml.safe_load(Path(path).read_text()) or {}; allowed=set(SightlineConfig.__dataclass_fields__)
    unknown=set(raw)-allowed
    if unknown: raise ValueError(f'unknown Sightline config keys: {sorted(unknown)}')
    required=allowed-set(raw)
    if required: raise ValueError(f'missing Sightline config keys: {sorted(required)}')
    if tuple(raw['history_sizes'])!=(16,2,1) or raw['chunk_length']!=33 or raw['chunk_stride']!=32: raise ValueError('invalid causal history/chunk semantics')
    if not (raw['ray_epsilon']>0 and 0<=raw['scale_augmentation_probability']<=1): raise ValueError('invalid ray/augmentation values')
    if raw['scale_augmentation_range'][0]>=raw['scale_augmentation_range'][1] or raw['lora_rank'] not in (8,16) or raw['memory_pool']!=2 or raw['memory_budget']<1: raise ValueError('invalid memory/LoRA configuration')
    for key in ('sightline_layers','memory_layers','correspondence_layers','lora_layers'):
        if any(int(x)<0 for x in raw[key]): raise ValueError(f'invalid {key}')
    if not (0<=raw['lambda_corr_decay_start']<=1) or raw['diagnostics_frequency']<1: raise ValueError('invalid schedule/diagnostic configuration')
    if raw['phase'] not in ('P1','P2','P3') or raw['chunk_count'] not in range(1,7) or tuple(raw['pyramid_steps'])!=(2,2,2): raise ValueError('invalid phase/chunk/stage configuration')
    if (raw['p1_steps'],raw['p2_steps'],raw['p3_steps'])!=(400,600,1500) or raw['ddp_world_size'] != 4 or raw['checkpoint_every']!=100 or raw['diagnostics_frequency']!=5: raise ValueError('formal schedule must be P1/P2/P3=400/600/1500, DDP=4, metrics=5, checkpoint=100')
    if raw['source_height']<=0 or raw['source_width']<=0 or raw['accumulation_steps']<1 or raw['correspondence_rows_per_batch']<1: raise ValueError('invalid data/training configuration')
    return SightlineConfig(**{k:(tuple(v) if k.endswith('_layers') or k=='history_sizes' else tuple(v) if k=='scale_augmentation_range' else v) for k,v in raw.items()})
