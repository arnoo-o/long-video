"""Strict Sightline checkpoint metadata and resume validation."""
from __future__ import annotations
import hashlib,json,random
from pathlib import Path
import torch
SEMANTICS='sightline-v1'; SCHEMA='sightline-checkpoint-v2'
def config_fingerprint(config): return hashlib.sha256(json.dumps(config,sort_keys=True,default=str).encode()).hexdigest()
def runtime_provenance(pipe, model_id, helios_root):
    root=Path(helios_root); transformer=root/'helios/diffusers_version/transformer_helios_diffusers.py'; pipeline=root/'helios/diffusers_version/pipeline_helios_diffusers.py'
    if not transformer.is_file() or not pipeline.is_file(): raise FileNotFoundError('pinned Helios source files are required for provenance')
    model_path=Path(model_id); identity_file=model_path/'model_index.json'
    model_identity=hashlib.sha256(identity_file.read_bytes()).hexdigest() if identity_file.is_file() else hashlib.sha256(str(model_id).encode()).hexdigest()
    scheduler_config=dict(pipe.scheduler.config)
    return {'transformer_source_sha256':hashlib.sha256(transformer.read_bytes()).hexdigest(),'pipeline_source_sha256':hashlib.sha256(pipeline.read_bytes()).hexdigest(),'scheduler_class':type(pipe.scheduler).__module__+'.'+type(pipe.scheduler).__qualname__,'scheduler_config_sha256':config_fingerprint(scheduler_config),'model_id':str(model_id),'model_identity':model_identity}
def save_checkpoint(path, model, optimizer, scheduler, step, *, config, helios_fingerprint, layers, memory_config):
    payload={'model':model.state_dict(),'optimizer':optimizer.state_dict() if optimizer else None,'scheduler':scheduler.state_dict() if scheduler else None,'step':int(step),'rng_torch':torch.get_rng_state(),'rng_python':random.getstate(),'sightline_training_semantics_version':SEMANTICS,'sightline_checkpoint_schema_version':SCHEMA,'config':config,'config_fingerprint':config_fingerprint(config),'helios_fingerprint':helios_fingerprint,'layers':list(layers),'memory_config':memory_config}
    Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,path)
def validate_checkpoint(payload, *, config, helios_fingerprint, layers, memory_config):
    if payload.get('sightline_training_semantics_version')!=SEMANTICS or payload.get('sightline_checkpoint_schema_version')!=SCHEMA: raise RuntimeError('stale Sightline checkpoint semantics')
    if payload.get('helios_fingerprint')!=helios_fingerprint or payload.get('config_fingerprint')!=config_fingerprint(config): raise RuntimeError('Sightline checkpoint provenance/config mismatch')
    if tuple(payload.get('layers',()))!=tuple(layers) or payload.get('memory_config')!=memory_config: raise RuntimeError('Sightline checkpoint layer/memory mismatch')

def save_runtime_checkpoint(path, trainable, memory, transformer, optimizer, scheduler, step, *, config, helios_fingerprint, layers, memory_config, provenance=None):
    lora={name:value for name,value in transformer.state_dict().items() if 'lora_' in name}
    payload={'trainable':trainable.state_dict(),'memory':memory.state_dict(),'lora':lora,
        'optimizer':optimizer.state_dict() if optimizer else None,'scheduler':scheduler.state_dict() if scheduler else None,
        'step':int(step),'rng_torch':torch.get_rng_state(),'rng_python':random.getstate(),
        'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'sightline_training_semantics_version':SEMANTICS,'sightline_checkpoint_schema_version':SCHEMA,
        'config':config,'config_fingerprint':config_fingerprint(config),'helios_fingerprint':helios_fingerprint,
        'layers':list(layers),'memory_config':memory_config,'runtime_provenance':provenance}
    Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,path)

def restore_runtime_checkpoint(payload, trainable, memory, transformer, *, config, helios_fingerprint, layers, memory_config, optimizer=None, scheduler=None, restore_rng=False, provenance=None):
    validate_checkpoint(payload,config=config,helios_fingerprint=helios_fingerprint,layers=layers,memory_config=memory_config)
    if provenance is not None and payload.get('runtime_provenance')!=provenance: raise RuntimeError('Sightline checkpoint runtime provenance mismatch')
    trainable.load_state_dict(payload['trainable'],strict=True); memory.load_state_dict(payload['memory'],strict=True)
    missing,unexpected=transformer.load_state_dict(payload.get('lora',{}),strict=False)
    unexpected=[name for name in unexpected if 'lora_' in name]
    expected={name for name,_ in transformer.named_parameters() if 'lora_' in name}
    if unexpected or not expected.issubset(payload.get('lora',{})): raise RuntimeError('checkpoint LoRA parameter set mismatch')
    if optimizer is not None:
        if payload.get('optimizer') is None: raise RuntimeError('checkpoint has no optimizer state')
        optimizer.load_state_dict(payload['optimizer'])
    if scheduler is not None:
        if payload.get('scheduler') is None: raise RuntimeError('checkpoint has no scheduler state')
        scheduler.load_state_dict(payload['scheduler'])
    if restore_rng:
        torch.set_rng_state(payload['rng_torch']); random.setstate(payload['rng_python'])
        if torch.cuda.is_available() and payload.get('rng_cuda') is not None: torch.cuda.set_rng_state_all(payload['rng_cuda'])
    return int(payload['step'])
