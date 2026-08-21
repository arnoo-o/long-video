"""Strict Sightline checkpoint metadata and resume validation."""
from __future__ import annotations
import hashlib,json,random
from pathlib import Path
import torch
SEMANTICS='sightline-v1'; SCHEMA='sightline-checkpoint-v1'
def config_fingerprint(config): return hashlib.sha256(json.dumps(config,sort_keys=True,default=str).encode()).hexdigest()
def save_checkpoint(path, model, optimizer, scheduler, step, *, config, helios_fingerprint, layers, memory_config):
    payload={'model':model.state_dict(),'optimizer':optimizer.state_dict() if optimizer else None,'scheduler':scheduler.state_dict() if scheduler else None,'step':int(step),'rng_torch':torch.get_rng_state(),'rng_python':random.getstate(),'sightline_training_semantics_version':SEMANTICS,'sightline_checkpoint_schema_version':SCHEMA,'config':config,'config_fingerprint':config_fingerprint(config),'helios_fingerprint':helios_fingerprint,'layers':list(layers),'memory_config':memory_config}
    Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,path)
def validate_checkpoint(payload, *, config, helios_fingerprint, layers, memory_config):
    if payload.get('sightline_training_semantics_version')!=SEMANTICS or payload.get('sightline_checkpoint_schema_version')!=SCHEMA: raise RuntimeError('stale Sightline checkpoint semantics')
    if payload.get('helios_fingerprint')!=helios_fingerprint or payload.get('config_fingerprint')!=config_fingerprint(config): raise RuntimeError('Sightline checkpoint provenance/config mismatch')
    if tuple(payload.get('layers',()))!=tuple(layers) or payload.get('memory_config')!=memory_config: raise RuntimeError('Sightline checkpoint layer/memory mismatch')

def save_runtime_checkpoint(path, trainable, memory, transformer, optimizer, scheduler, step, *, config, helios_fingerprint, layers, memory_config):
    lora={name:value for name,value in transformer.state_dict().items() if 'lora_' in name}
    payload={'trainable':trainable.state_dict(),'memory':memory.state_dict(),'lora':lora,
        'optimizer':optimizer.state_dict() if optimizer else None,'scheduler':scheduler.state_dict() if scheduler else None,
        'step':int(step),'sightline_training_semantics_version':SEMANTICS,'sightline_checkpoint_schema_version':SCHEMA,
        'config':config,'config_fingerprint':config_fingerprint(config),'helios_fingerprint':helios_fingerprint,
        'layers':list(layers),'memory_config':memory_config}
    Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,path)

def restore_runtime_checkpoint(payload, trainable, memory, transformer, *, config, helios_fingerprint, layers, memory_config):
    validate_checkpoint(payload,config=config,helios_fingerprint=helios_fingerprint,layers=layers,memory_config=memory_config)
    trainable.load_state_dict(payload['trainable'],strict=True); memory.load_state_dict(payload['memory'],strict=True)
    missing,unexpected=transformer.load_state_dict(payload.get('lora',{}),strict=False)
    unexpected=[name for name in unexpected if 'lora_' in name]
    expected={name for name,_ in transformer.named_parameters() if 'lora_' in name}
    if unexpected or not expected.issubset(payload.get('lora',{})): raise RuntimeError('checkpoint LoRA parameter set mismatch')
    return int(payload['step'])
