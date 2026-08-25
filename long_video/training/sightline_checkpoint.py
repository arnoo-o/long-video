"""Strict Sightline checkpoint metadata and resume validation."""
from __future__ import annotations
import hashlib,io,json,random
import numpy as np
from pathlib import Path
import torch
SEMANTICS='sightline-v9'; SCHEMA='sightline-checkpoint-v11'
def config_fingerprint(config): return hashlib.sha256(json.dumps(config,sort_keys=True,default=str).encode()).hexdigest()
def scheduler_config_fingerprint(config):
    config=dict(config)
    # Diffusers builds this field from a set, so its list order changes between
    # processes even though the scheduler configuration is identical.
    if '_use_default_values' in config: config['_use_default_values']=sorted(config['_use_default_values'])
    return config_fingerprint(config)
def _file_sha(path): return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
def runtime_provenance(pipe, model_id, helios_root, model_revision=None):
    root=Path(helios_root); transformer=root/'helios/diffusers_version/transformer_helios_diffusers.py'; pipeline=root/'helios/diffusers_version/pipeline_helios_diffusers.py'
    if not transformer.is_file() or not pipeline.is_file(): raise FileNotFoundError('pinned Helios source files are required for provenance')
    model_path=Path(model_id); local=model_path.is_dir(); transformer_config=config_fingerprint(dict(pipe.transformer.config))
    if local:
        model_index=_file_sha(model_path/'model_index.json')
        if model_index is None: raise RuntimeError('local model provenance requires model_index.json')
        weight_indices=sorted(model_path.rglob('*.safetensors.index.json'))+sorted(model_path.rglob('*.bin.index.json'))
        if weight_indices:
            weight_identity=[(str(path.relative_to(model_path)),_file_sha(path)) for path in weight_indices]
            weight_identity_kind='weight_index'
        else:
            weight_files=sorted((*model_path.rglob('*.safetensors'),*model_path.rglob('*.bin')))
            if not weight_files: raise RuntimeError('local model provenance found no weight index or weight files')
            weight_identity=[(str(path.relative_to(model_path)),path.stat().st_size) for path in weight_files]
            weight_identity_kind='weight_layout'
        weight_fingerprint=config_fingerprint(weight_identity)
        model_identity={'kind':'local','model_index_sha256':model_index,'transformer_config_sha256':transformer_config,'weight_identity_kind':weight_identity_kind,'weight_identity_sha256':weight_fingerprint}
    else:
        revision=model_revision or getattr(pipe.config,'_commit_hash',None)
        if not revision: raise RuntimeError('HF model provenance requires an explicit revision/commit')
        model_identity={'kind':'huggingface','revision':str(revision),'transformer_config_sha256':transformer_config}
    scheduler_config=dict(pipe.scheduler.config)
    return {'transformer_source_sha256':hashlib.sha256(transformer.read_bytes()).hexdigest(),'pipeline_source_sha256':hashlib.sha256(pipeline.read_bytes()).hexdigest(),'scheduler_class':type(pipe.scheduler).__module__+'.'+type(pipe.scheduler).__qualname__,'scheduler_config_sha256':scheduler_config_fingerprint(scheduler_config),'model_id':str(model_id),'model_identity':model_identity}

def _provenance_matches(saved, current):
    """Permit relocating an identical local model while preserving strict fingerprints."""
    if not isinstance(saved,dict) or not isinstance(current,dict): return False
    keys=('transformer_source_sha256','pipeline_source_sha256','scheduler_class','scheduler_config_sha256','model_identity')
    return all(saved.get(key)==current.get(key) for key in keys)
def save_checkpoint(path, model, optimizer, scheduler, step, *, config, helios_fingerprint, layers, memory_config):
    payload={'model':model.state_dict(),'optimizer':optimizer.state_dict() if optimizer else None,'scheduler':scheduler.state_dict() if scheduler else None,'step':int(step),'rng_torch':torch.get_rng_state(),'rng_python':random.getstate(),'sightline_training_semantics_version':SEMANTICS,'sightline_checkpoint_schema_version':SCHEMA,'config':config,'config_fingerprint':config_fingerprint(config),'helios_fingerprint':helios_fingerprint,'layers':list(layers),'memory_config':memory_config}
    Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,path)
def validate_checkpoint(payload, *, config, helios_fingerprint, layers, memory_config, allow_memory_layer_migration=False):
    if payload.get('sightline_training_semantics_version')!=SEMANTICS or payload.get('sightline_checkpoint_schema_version')!=SCHEMA:
        raise RuntimeError(f'incompatible Sightline checkpoint: expected {SEMANTICS}/{SCHEMA} for padded Q/K-only Sightline and rank-16 LoRA; got {payload.get("sightline_training_semantics_version")}/{payload.get("sightline_checkpoint_schema_version")}')
    if payload.get('helios_fingerprint')!=helios_fingerprint: raise RuntimeError('Sightline checkpoint provenance mismatch')
    saved_config=payload.get('config',{})
    config_match=payload.get('config_fingerprint')==config_fingerprint(config)
    if allow_memory_layer_migration and not config_match:
        old=dict(saved_config); new=dict(config); old.pop('memory_layers',None); new.pop('memory_layers',None)
        config_match=old==new
    if not config_match: raise RuntimeError('Sightline checkpoint config mismatch')
    if tuple(payload.get('layers',()))!=tuple(layers): raise RuntimeError('Sightline checkpoint layer mismatch')
    if not allow_memory_layer_migration and payload.get('memory_config')!=memory_config: raise RuntimeError('Sightline checkpoint memory mismatch')

def _numpy_rng_state():
    kind, values, position, has_gauss, cached = np.random.get_state()
    return (str(kind), torch.as_tensor(values, dtype=torch.uint32), int(position), int(has_gauss), float(cached))

def _restore_numpy_rng_state(state):
    kind, values, position, has_gauss, cached = state
    np.random.set_state((str(kind), torch.as_tensor(values, dtype=torch.uint32).cpu().numpy(), int(position), int(has_gauss), float(cached)))

def capture_rng_state():
    return {'torch':torch.get_rng_state(),'python':random.getstate(),'numpy':_numpy_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def serialize_rng_state(state) -> torch.Tensor:
    buffer=io.BytesIO(); torch.save(state,buffer)
    return torch.frombuffer(bytearray(buffer.getvalue()),dtype=torch.uint8).clone()

def deserialize_rng_state(value:torch.Tensor):
    return torch.load(io.BytesIO(value.detach().cpu().numpy().tobytes()),map_location='cpu',weights_only=False)

def gather_rank_rng_states(world_size:int,device) -> list[torch.Tensor]:
    """All-gather variable-length serialized states without object collectives."""
    local=serialize_rng_state(capture_rng_state()).to(device)
    if world_size==1:return [local.cpu()]
    if not torch.distributed.is_initialized():raise RuntimeError('RNG all-gather requires an initialized process group')
    length=torch.tensor([local.numel()],device=device,dtype=torch.int64); lengths=[torch.zeros_like(length) for _ in range(world_size)]
    torch.distributed.all_gather(lengths,length); maximum=max(int(value.item()) for value in lengths)
    padded=torch.zeros(maximum,device=device,dtype=torch.uint8); padded[:local.numel()]=local
    gathered=[torch.empty_like(padded) for _ in range(world_size)]; torch.distributed.all_gather(gathered,padded)
    return [value[:int(length.item())].cpu() for value,length in zip(gathered,lengths)]

def _restore_rng_state(state):
    torch.set_rng_state(state['torch']); random.setstate(state['python']); _restore_numpy_rng_state(state['numpy'])
    if torch.cuda.is_available() and state.get('cuda') is not None: torch.cuda.set_rng_state_all(state['cuda'])

def save_runtime_checkpoint(path, trainable, memory, transformer, optimizer, scheduler, step, *, config, helios_fingerprint, layers, memory_config, provenance=None, rng_states=None, world_size=1):
    if rng_states is None:
        if int(world_size)!=1: raise RuntimeError('multi-rank checkpoints require explicitly gathered RNG states')
        rng_states=[serialize_rng_state(capture_rng_state())]
    lora={name:value for name,value in transformer.state_dict().items() if 'lora_' in name}
    payload={'trainable':trainable.state_dict(),'memory':memory.state_dict(),'lora':lora,
        'optimizer':optimizer.state_dict() if optimizer else None,'scheduler':scheduler.state_dict() if scheduler else None,
        'step':int(step),'rng_torch':torch.get_rng_state(),'rng_python':random.getstate(),'rng_numpy':_numpy_rng_state(),
        'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,'rng_states':rng_states,'rng_world_size':int(world_size),
        'sightline_training_semantics_version':SEMANTICS,'sightline_checkpoint_schema_version':SCHEMA,
        'config':config,'config_fingerprint':config_fingerprint(config),'helios_fingerprint':helios_fingerprint,
        'layers':list(layers),'memory_config':memory_config,'runtime_provenance':provenance}
    Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,path)

def restore_runtime_checkpoint(payload, trainable, memory, transformer, *, config, helios_fingerprint, layers, memory_config, optimizer=None, scheduler=None, restore_rng=False, provenance=None, rank=0, world_size=1, allow_memory_layer_migration=False):
    validate_checkpoint(payload,config=config,helios_fingerprint=helios_fingerprint,layers=layers,memory_config=memory_config,allow_memory_layer_migration=allow_memory_layer_migration)
    if provenance is not None and not _provenance_matches(payload.get('runtime_provenance'),provenance): raise RuntimeError('Sightline checkpoint runtime provenance mismatch')
    trainable.load_state_dict(payload['trainable'],strict=True)
    if not allow_memory_layer_migration: memory.load_state_dict(payload['memory'],strict=True)
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
        states=payload.get('rng_states')
        saved_world=int(payload.get('rng_world_size',-1))
        if states is None or saved_world<1:raise RuntimeError('checkpoint has no per-rank serialized RNG states')
        if saved_world!=int(world_size):raise RuntimeError(f'checkpoint RNG world size {saved_world} != runtime world size {world_size}; use an explicit deterministic reseed workflow')
        if len(states)!=saved_world or not 0<=int(rank)<saved_world:raise RuntimeError('checkpoint per-rank RNG state set is incomplete')
        _restore_rng_state(deserialize_rng_state(states[int(rank)]))
    return int(payload['step'])
