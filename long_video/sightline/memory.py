"""Chunk-atomic K/V memory with a fixed anchor and geometry retrieval."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn


def _pinned_cpu(value:torch.Tensor) -> torch.Tensor:
    value=value.detach().to(device='cpu').contiguous()
    return value.pin_memory() if torch.cuda.is_available() and not value.is_pinned() else value


@dataclass
class MemoryToken:
    chunk_index:int; token_index:int; temporal:int; pooled_y:int; pooled_x:int


@dataclass
class MemoryChunk:
    chunk_id:int
    hidden:torch.Tensor
    rays:torch.Tensor
    temporal:torch.Tensor
    pooled_y:torch.Tensor
    pooled_x:torch.Tensor
    camera_poses:torch.Tensor
    global_latent_ids:torch.Tensor
    timestamp:int

    def __post_init__(self):
        count=int(self.hidden.shape[1])
        if self.hidden.ndim!=3 or self.rays.ndim!=3 or self.rays.shape[:2]!=self.hidden.shape[:2] or self.rays.shape[-1]!=7:
            raise ValueError('MemoryChunk hidden/rays must be [B,N,D] and [B,N,7]')
        if any(value.ndim!=1 or value.numel()!=count for value in (self.temporal,self.pooled_y,self.pooled_x,self.global_latent_ids)):
            raise ValueError('MemoryChunk metadata must match its token axis')
        if not self.hidden.is_contiguous() or not self.rays.is_contiguous():
            raise ValueError('MemoryChunk tensors must be contiguous')

    @property
    def token_count(self): return int(self.hidden.shape[1])

    @property
    def tokens(self):
        return tuple(MemoryToken(self.chunk_id,index,int(self.temporal[index]),int(self.pooled_y[index]),int(self.pooled_x[index])) for index in range(self.token_count))


def _trajectory_score(query:torch.Tensor,history:torch.Tensor,tau_pos:float,tau_angle:float)->float:
    query=query.detach().float().cpu(); history=history.detach().float().cpu()
    if query.ndim==4: query=query[0]
    if history.ndim==4: history=history[0]
    qpos=query[:,:3,3]; hpos=history[:,:3,3]
    qforward=query[:,:3,2]; hforward=history[:,:3,2]
    distances=torch.cdist(qpos,hpos)
    angles=torch.acos(torch.einsum('id,jd->ij',qforward,hforward).clamp(-1,1))
    pair_scores=torch.exp(-distances/tau_pos)*torch.exp(-angles/tau_angle)
    return float(pair_scores.max())


def select_memory_chunks(archive:dict[int,MemoryChunk],*,query_chunk:int,query_camera_poses:torch.Tensor,
                         native_history_chunk_ids=(),tau_pos:float=1.0,tau_angle:float=0.78539816339,
                         retrieval_count:int=3)->tuple[MemoryChunk,...]:
    """Select only past chunks using known camera trajectories; never duplicates."""
    if tau_pos<=0 or tau_angle<=0 or retrieval_count<0: raise ValueError('invalid Memory retrieval configuration')
    # Native history and long-term scene Memory are independent representations.
    # A past chunk may legally appear in both attention paths.
    past={chunk_id:chunk for chunk_id,chunk in archive.items() if chunk_id<query_chunk}
    selected=[]
    if 0 in past: selected.append(past[0])
    excluded={0,query_chunk}
    candidates=[chunk for chunk_id,chunk in past.items() if chunk_id not in excluded]
    candidates.sort(key=lambda chunk:(-_trajectory_score(query_camera_poses,chunk.camera_poses,tau_pos,tau_angle),chunk.chunk_id))
    selected.extend(candidates[:retrieval_count])
    if len({chunk.chunk_id for chunk in selected})!=len(selected): raise RuntimeError('Memory selector returned a duplicate chunk')
    return tuple(selected)


class LongTermKVMemory:
    def __init__(self,budget=13312,pool=2,*,tau_pos=1.0,tau_angle=0.78539816339):
        self.budget=int(budget); self.pool=int(pool); self.tau_pos=float(tau_pos); self.tau_angle=float(tau_angle)
        self.archive:dict[int,MemoryChunk]={}; self.timestamp=None; self.rope=None; self.enabled=True; self.last_active_tokens=[]; self.last_selected_chunk_ids=()
        self._active_query_chunk=None; self._active_hidden=None; self._active_rays=None; self._active_tokens=()
        self._host_hidden=None; self._host_rays=None
        self._active_temporal=self._active_y=self._active_x=self._active_global_ids=self._active_chunk_ids=None
        self._host_temporal=self._host_y=self._host_x=self._host_global_ids=self._host_chunk_ids=None
        self._active_identity_metadata=()
        self.coordinator=None; self.layer=None
        self.protected_archive_chunk_ids=frozenset((0,))

    @property
    def tokens(self): return [token for chunk in self.archive.values() for token in chunk.tokens]

    def capture(self,hidden,rays,chunk_index,*,grid_shape=None,ray_recompute=None,camera_poses=None,timestamp=None):
        if not self.enabled: return
        if hidden.ndim<3 or rays.shape[:2]!=hidden.shape[:2]: raise ValueError('memory hidden/ray token count mismatch')
        if camera_poses is None or camera_poses.ndim not in (3,4): raise ValueError('Memory chunk requires its known camera trajectory')
        T,H,W=grid_shape or (0,0,0)
        if T*H*W!=hidden.shape[1]: raise ValueError('memory grid does not match hidden/rays')
        if H%self.pool or W%self.pool: raise ValueError('memory grid must be divisible by spatial pooling')
        hh=hidden.reshape(hidden.shape[0],T,H,W,-1); rr=rays.reshape(rays.shape[0],T,H,W,-1)
        hh=hh.reshape(hh.shape[0],T,H//self.pool,self.pool,W//self.pool,self.pool,-1).mean((3,5))
        rr=rr.reshape(rr.shape[0],T,H//self.pool,self.pool,W//self.pool,self.pool,-1).mean((3,5))
        if ray_recompute is not None: rr=ray_recompute((T,H//self.pool,W//self.pool)).reshape_as(rr)
        pooled_h,pooled_w=H//self.pool,W//self.pool
        return self.capture_pooled(hh[:,1:].reshape(hh.shape[0],-1,hh.shape[-1]),rr[:,1:].reshape(rr.shape[0],-1,rr.shape[-1]),chunk_index,grid_shape=(T,pooled_h,pooled_w),camera_poses=camera_poses,timestamp=timestamp)

    def capture_pooled(self,hidden,rays,chunk_index,*,grid_shape,camera_poses=None,timestamp=None):
        """Archive already pooled temporal1..T-1 tensors without pooling twice."""
        if not self.enabled:return
        if camera_poses is None or camera_poses.ndim not in (3,4): raise ValueError('Memory chunk requires its known camera trajectory')
        T,pooled_h,pooled_w=map(int,grid_shape); expected=(T-1)*pooled_h*pooled_w
        if hidden.ndim!=3 or rays.ndim!=3 or hidden.shape[:2]!=rays.shape[:2] or hidden.shape[1]!=expected or rays.shape[-1]!=7:
            raise ValueError('pooled Memory hidden/rays do not match temporal1 grid')
        rr=rays.clone()
        rr[...,:3]=rr[...,:3]/rr[...,:3].norm(dim=-1,keepdim=True).clamp_min(1e-6)
        rr[...,3:6]=rr[...,3:6]/rr[...,3:6].norm(dim=-1,keepdim=True).clamp_min(1e-6)
        hidden_chunk=_pinned_cpu(hidden); ray_chunk=_pinned_cpu(rr)
        temporal=_pinned_cpu(torch.arange(1,T,dtype=torch.long).repeat_interleave(pooled_h*pooled_w))
        pooled_y=_pinned_cpu(torch.arange(pooled_h,dtype=torch.long).repeat_interleave(pooled_w).repeat(T-1))
        pooled_x=_pinned_cpu(torch.arange(pooled_w,dtype=torch.long).repeat(pooled_h*(T-1)))
        global_ids=_pinned_cpu(int(chunk_index)*8+temporal)
        chunk=MemoryChunk(int(chunk_index),hidden_chunk,ray_chunk,temporal,pooled_y,pooled_x,_pinned_cpu(camera_poses),global_ids,int(chunk_index if timestamp is None else timestamp))
        chunk_id=int(chunk_index)
        # A repeated capture must never silently replace the permanent anchor.
        if chunk_id in self.protected_archive_chunk_ids and chunk_id in self.archive:
            raise RuntimeError('permanent Memory anchor chunk0 cannot be overwritten')
        self.archive[chunk_id]=chunk

    def evict_archive_chunk(self,chunk_id:int) -> bool:
        """Ordinary archive eviction; the permanent chunk0 anchor is protected."""
        chunk_id=int(chunk_id)
        if chunk_id in self.protected_archive_chunk_ids:return False
        return self.archive.pop(chunk_id,None) is not None

    def select_chunks(self,current_chunk,query_camera_poses,native_history_chunk_ids=()):
        return select_memory_chunks(self.archive,query_chunk=int(current_chunk),query_camera_poses=query_camera_poses,native_history_chunk_ids=native_history_chunk_ids,tau_pos=self.tau_pos,tau_angle=self.tau_angle)

    def active_tokens(self,current_global_start=None,*,query_camera_poses=None,native_history_chunk_ids=()):
        current_chunk=int(current_global_start//8) if current_global_start is not None else (max(self.archive,default=-1)+1)
        if query_camera_poses is None:
            if current_chunk in self.archive: query_camera_poses=self.archive[current_chunk].camera_poses
            elif self.archive: query_camera_poses=self.archive[max(self.archive)].camera_poses
            else: return []
        chunks=self.select_chunks(current_chunk,query_camera_poses,native_history_chunk_ids)
        tokens=[token for chunk in chunks for token in chunk.tokens]
        if len(tokens)>self.budget: raise RuntimeError(f'complete active Memory chunks exceed token budget: {len(tokens)}>{self.budget}')
        self.last_active_tokens=tokens; self.last_selected_chunk_ids=tuple(chunk.chunk_id for chunk in chunks)
        return tokens

    def prepare_active_memory(self, *, query_chunk:int, query_camera_poses:torch.Tensor,
                              native_history_chunk_ids=(), device=None, dtype=None, selected_chunk_ids=None,
                              identity_metadata=None, shared_host_metadata=None):
        """Select and materialize the complete active chunks once per query chunk."""
        query_chunk=int(query_chunk)
        if self._active_query_chunk == query_chunk: return self.last_selected_chunk_ids
        chunks=self.select_chunks(query_chunk,query_camera_poses) if selected_chunk_ids is None else tuple(self.archive[int(chunk_id)] for chunk_id in selected_chunk_ids)
        token_count=sum(chunk.token_count for chunk in chunks)
        if token_count>self.budget:
            raise RuntimeError(f'complete active Memory chunks exceed token budget: {token_count}>{self.budget}')
        self.last_active_tokens=[]; self.last_selected_chunk_ids=tuple(chunk.chunk_id for chunk in chunks)
        self._active_query_chunk=query_chunk; self._active_tokens=()
        self._active_hidden=self._active_rays=None
        self._active_temporal=self._active_y=self._active_x=self._active_global_ids=self._active_chunk_ids=None
        if chunks:
            # Assemble pinned host packs only. The layer coordinator moves at
            # most two packs to CUDA using its rolling prefetch buffers.
            self._host_hidden=_pinned_cpu(torch.cat([chunk.hidden for chunk in chunks],1))
            if shared_host_metadata is None:
                self._host_rays=_pinned_cpu(torch.cat([chunk.rays for chunk in chunks],1))
                self._host_temporal=_pinned_cpu(torch.cat([chunk.temporal for chunk in chunks]))
                self._host_y=_pinned_cpu(torch.cat([chunk.pooled_y for chunk in chunks]))
                self._host_x=_pinned_cpu(torch.cat([chunk.pooled_x for chunk in chunks]))
                self._host_global_ids=_pinned_cpu(torch.cat([chunk.global_latent_ids for chunk in chunks]))
                self._host_chunk_ids=_pinned_cpu(torch.cat([torch.full((chunk.token_count,),chunk.chunk_id,dtype=torch.long) for chunk in chunks]))
            else:
                (self._host_rays,self._host_temporal,self._host_y,self._host_x,
                 self._host_global_ids,self._host_chunk_ids)=shared_host_metadata
                if self._host_rays.shape[1]!=token_count or any(value.numel()!=token_count for value in shared_host_metadata[1:]):
                    raise RuntimeError('shared Memory identity metadata differs across layers')
            # Build Python identity data once from CPU archive metadata.  The
            # processor reuses this cache; no GPU tensor is converted to a list
            # while denoising or attending.
            self._active_identity_metadata=tuple(
                (int(global_id),int(y),int(x))
                for chunk in chunks
                for global_id,y,x in zip(chunk.global_latent_ids.tolist(),chunk.pooled_y.tolist(),chunk.pooled_x.tolist())
            ) if identity_metadata is None else tuple(identity_metadata)
        else:
            self._host_hidden=self._host_rays=None
            self._host_temporal=self._host_y=self._host_x=self._host_global_ids=self._host_chunk_ids=None
            self._active_identity_metadata=()
        return self.last_selected_chunk_ids

    def clear_active_memory(self):
        self._active_query_chunk=None; self._active_hidden=None; self._active_rays=None; self._active_tokens=()
        self._host_hidden=self._host_rays=None
        self._active_temporal=self._active_y=self._active_x=self._active_global_ids=self._active_chunk_ids=None
        self._host_temporal=self._host_y=self._host_x=self._host_global_ids=self._host_chunk_ids=None
        self._active_identity_metadata=()
        self.last_active_tokens=[]; self.last_selected_chunk_ids=()

    def get(self,current_global_start=None,*,query_camera_poses=None,native_history_chunk_ids=(),device=None,dtype=None):
        current_chunk=int(current_global_start//8) if current_global_start is not None else None
        if current_chunk is None: raise ValueError('Memory get requires current_global_start')
        if self._active_query_chunk != current_chunk:
            if query_camera_poses is None:return None,None
            self.prepare_active_memory(query_chunk=current_chunk,query_camera_poses=query_camera_poses,device=device,dtype=dtype)
        if self._host_hidden is None:return None,None
        if self.coordinator is not None and torch.device(device).type=='cuda':
            self.coordinator.acquire_layer(self.layer,device=device,dtype=dtype)
        else:
            self._active_hidden=self._host_hidden.to(device=device,dtype=dtype,non_blocking=True)
            self._active_rays=self._host_rays.to(device=device,dtype=dtype,non_blocking=True)
            self._active_temporal=self._host_temporal.to(device=device,non_blocking=True)
            self._active_y=self._host_y.to(device=device,non_blocking=True); self._active_x=self._host_x.to(device=device,non_blocking=True)
            self._active_global_ids=self._host_global_ids.to(device=device,non_blocking=True); self._active_chunk_ids=self._host_chunk_ids.to(device=device,non_blocking=True)
        return self._active_hidden,self._active_rays

    def __len__(self): return sum(chunk.token_count for chunk in self.archive.values())

    def position_metadata(self,tokens,device,current_global_start):
        if self._active_global_ids is not None:
            global_ids=self._active_global_ids.to(device=device)
            y_ids=self._active_y.to(device=device); x_ids=self._active_x.to(device=device)
        elif tokens:
            global_ids=torch.tensor([token.chunk_index*8+token.temporal for token in tokens],device=device)
            y_ids=torch.tensor([token.pooled_y for token in tokens],device=device); x_ids=torch.tensor([token.pooled_x for token in tokens],device=device)
        else:return None
        frame=(19-(int(current_global_start)-global_ids)).clamp(1,18).to(torch.float32)
        offset=(self.pool-1)/2
        y=y_ids.to(torch.float32)*self.pool+offset; x=x_ids.to(torch.float32)*self.pool+offset
        return frame.view(1,-1,1,1),y.view(1,-1,1,1),x.view(1,-1,1,1)

    def memory_rotary_emb(self,tokens,device,current_global_start):
        if self.rope is None: raise RuntimeError('Helios native RoPE module is not bound to memory')
        positions=self.position_metadata(tokens,device,current_global_start)
        if positions is None:return None
        return self.rope.forward_with_positions(*positions,device=device).flatten(2).transpose(1,2)

    def append_native_attention(self,attn,key,value,rotary_emb,rotary_apply,*,current_chunk,current_global_start,timestamp_embedding=None,sightline_projector=None,scale_delta=None,query_camera_poses=None,native_history_chunk_ids=(),**kwargs):
        hidden,rays=self.get(current_global_start,query_camera_poses=query_camera_poses,native_history_chunk_ids=native_history_chunk_ids,device=key.device,dtype=key.dtype)
        if not self.enabled or hidden is None:return key,value,{'memory_tokens':0,'memory_chunk_ids':[]}
        if hidden.shape[0]!=key.shape[0]:
            if hidden.shape[0]==1:hidden=hidden.expand(key.shape[0],-1,-1);rays=rays.expand(key.shape[0],-1,-1)
            else:raise ValueError('memory batch differs from attention batch')
        mem_k=attn.norm_k(attn.to_k(hidden)).unflatten(2,(attn.heads,-1)); mem_v=attn.to_v(hidden).unflatten(2,(attn.heads,-1))
        memory_rotary=self.memory_rotary_emb(None,hidden.device,current_global_start).to(mem_k)
        if memory_rotary.shape[1]!=mem_k.shape[1]:raise RuntimeError('memory rotary embedding count mismatch')
        mem_k=rotary_apply(mem_k,memory_rotary)
        if sightline_projector is not None:
            delta=sightline_projector.project(rays.to(hidden),kind='k',training=sightline_projector.training,scale_delta=scale_delta)
            mem_k=mem_k+delta.unflatten(-1,(attn.heads,-1))
        if timestamp_embedding is not None:
            ages=(int(current_chunk)-self._active_chunk_ids).clamp(0,timestamp_embedding.num_embeddings-1)
            if timestamp_embedding.weight.device!=hidden.device or timestamp_embedding.weight.dtype!=hidden.dtype:raise RuntimeError('memory timestamp embedding device/dtype mismatch')
            mem_k=mem_k+timestamp_embedding(ages).to(mem_k.dtype).unsqueeze(0).unflatten(2,(attn.heads,-1))
        memory_type_embedding=getattr(self,'memory_type_embedding',None)
        if memory_type_embedding is not None:
            if memory_type_embedding.numel()!=mem_k.shape[2]*mem_k.shape[3]: raise RuntimeError('memory type embedding does not match Memory K head dimension')
            mem_k=mem_k+memory_type_embedding.to(mem_k).view(1,1,mem_k.shape[2],mem_k.shape[3])
        meta={'memory_tokens':mem_k.shape[1],'memory_chunk_count':len(self.last_selected_chunk_ids),'memory_chunk_ids':list(self.last_selected_chunk_ids),'memory_global_ids':tuple(global_id for global_id,_,_ in self._active_identity_metadata)}
        return torch.cat((key,mem_k),1),torch.cat((value,mem_v),1),meta

    def active_identity_metadata(self):
        return self._active_identity_metadata

    def _memory_rays(self):
        return torch.cat([chunk.rays for chunk in self.archive.values()],1) if self.archive else None


class LayerKVMemoryBank(nn.Module):
    def __init__(self,layers,budget=13312,pool=2,hidden_dim=None,*,tau_pos=1.0,tau_angle=0.78539816339):
        super().__init__(); self.banks={int(layer):LongTermKVMemory(budget,pool,tau_pos=tau_pos,tau_angle=tau_angle) for layer in layers}; self._selected_query_chunk=None; self._selected_chunk_ids=()
        self._ordered_layers=tuple(sorted(self.banks)); self._prefetch_stream=None; self._staging={}; self._staging_events={}; self._device=None; self._dtype=None; self._last_access_layer=None
        for layer,bank in self.banks.items(): bank.coordinator=self; bank.layer=layer
        self.timestamp=nn.Embedding(64,hidden_dim) if hidden_dim is not None else None
        self.memory_type_embedding=nn.Parameter(torch.zeros(hidden_dim)) if hidden_dim is not None else None
        if self.timestamp is not None:nn.init.zeros_(self.timestamp.weight)
    def for_layer(self,layer):
        if int(layer) not in self.banks:raise KeyError(f'memory layer {layer} is not selected')
        return self.banks[int(layer)]
    def bind_rope(self,rope):
        for bank in self.banks.values():
            bank.rope=rope;bank.timestamp=self.timestamp;bank.memory_type_embedding=self.memory_type_embedding
    def set_enabled(self,enabled):
        for bank in self.banks.values():bank.enabled=bool(enabled)
    def reset(self):
        for bank in self.banks.values():bank.archive.clear();bank.clear_active_memory()
        self._selected_query_chunk=None; self._selected_chunk_ids=(); self._clear_staging()
    def prepare_active_memory(self, *, query_chunk:int, query_camera_poses:torch.Tensor, native_history_chunk_ids=(), device=None, dtype=None):
        """Select once, then materialize the same complete chunks for every layer."""
        if not self.banks:return {}
        reference=next(iter(self.banks.values()))
        if self._selected_query_chunk==int(query_chunk): selected_ids=self._selected_chunk_ids
        else:
            selected_ids=tuple(chunk.chunk_id for chunk in reference.select_chunks(query_chunk,query_camera_poses))
            self._selected_query_chunk=int(query_chunk); self._selected_chunk_ids=selected_ids
        for bank in self.banks.values():
            if any(chunk_id not in bank.archive for chunk_id in selected_ids): raise RuntimeError('Memory layer archives differ in complete chunk membership')
        # Build identical pinned host packs for all layers. No Memory hidden or
        # ray tensor is materialized on CUDA until its processor is reached.
        reference.prepare_active_memory(query_chunk=query_chunk,query_camera_poses=query_camera_poses,
            device=device,dtype=dtype,selected_chunk_ids=selected_ids)
        identity_metadata=reference.active_identity_metadata()
        shared_host_metadata=(reference._host_rays,reference._host_temporal,reference._host_y,
                              reference._host_x,reference._host_global_ids,reference._host_chunk_ids)
        result={layer:reference.last_selected_chunk_ids for layer,bank in self.banks.items() if bank is reference}
        for layer,bank in self.banks.items():
            if bank is reference: continue
            result[layer]=bank.prepare_active_memory(query_chunk=query_chunk,query_camera_poses=query_camera_poses,
                device=device,dtype=dtype,selected_chunk_ids=selected_ids,identity_metadata=identity_metadata,
                shared_host_metadata=shared_host_metadata)
        self._clear_staging(); self._device=torch.device(device); self._dtype=dtype
        return result

    def _clear_bank_device_views(self,layer):
        bank=self.banks[int(layer)]
        bank._active_hidden=bank._active_rays=None
        bank._active_temporal=bank._active_y=bank._active_x=bank._active_global_ids=bank._active_chunk_ids=None

    def _clear_staging(self):
        for layer in tuple(getattr(self,'_staging',{})): self._clear_bank_device_views(layer)
        self._staging={}; self._staging_events={}; self._last_access_layer=None

    def _prefetch(self,layer):
        layer=int(layer)
        if layer in self._staging:return
        bank=self.banks[layer]
        if bank._host_hidden is None:return
        if self._device is None or self._device.type!='cuda':raise RuntimeError('Memory prefetch requires an active CUDA device')
        if self._prefetch_stream is None or self._prefetch_stream.device!=self._device:
            self._prefetch_stream=torch.cuda.Stream(device=self._device)
        with torch.cuda.stream(self._prefetch_stream):
            pack=(
                bank._host_hidden.to(self._device,dtype=self._dtype,non_blocking=True),
                bank._host_rays.to(self._device,dtype=self._dtype,non_blocking=True),
                bank._host_temporal.to(self._device,non_blocking=True),bank._host_y.to(self._device,non_blocking=True),
                bank._host_x.to(self._device,non_blocking=True),bank._host_global_ids.to(self._device,non_blocking=True),
                bank._host_chunk_ids.to(self._device,non_blocking=True),
            )
            event=torch.cuda.Event(); event.record(self._prefetch_stream)
        self._staging[layer]=pack; self._staging_events[layer]=event

    def _next_layer(self,layer):
        index=self._ordered_layers.index(int(layer))
        # Normal forward visits ascending layers. Checkpoint recompute visits
        # them in reverse; a repeated final layer marks that direction change.
        # Every new native Transformer pass starts at the minimum Memory layer,
        # including later pyramid/denoise stages after an ascending pass.
        if int(layer)==self._ordered_layers[0]:
            return self._ordered_layers[1] if len(self._ordered_layers)>1 else None
        if self._last_access_layer is not None and int(layer)<self._last_access_layer:
            return self._ordered_layers[index-1] if index else None
        if int(layer)==self._ordered_layers[-1] and self._last_access_layer==int(layer):
            return self._ordered_layers[index-1] if index else None
        return self._ordered_layers[index+1] if index+1<len(self._ordered_layers) else None

    def acquire_layer(self,layer,*,device=None,dtype=None):
        layer=int(layer); requested_device=torch.device(device)
        if requested_device!=self._device or dtype!=self._dtype:
            # _prepare_chunk may receive an FP32 scheduler/template latent,
            # while the actual Helios attention projection is BF16. Bind the
            # rolling buffers to the real K/V contract at first layer use.
            self._clear_staging(); self._device=requested_device; self._dtype=dtype
        self._prefetch(layer)
        event=self._staging_events.get(layer)
        if event is None:return
        torch.cuda.current_stream(self._device).wait_event(event)
        consumer_stream=torch.cuda.current_stream(self._device)
        pack=self._staging[layer]; bank=self.banks[layer]
        for value in pack:
            value.record_stream(consumer_stream)
        (bank._active_hidden,bank._active_rays,bank._active_temporal,bank._active_y,
         bank._active_x,bank._active_global_ids,bank._active_chunk_ids)=pack
        next_layer=self._next_layer(layer)
        keep={layer,next_layer}
        for stale in tuple(self._staging):
            if stale not in keep:
                self._staging.pop(stale,None); self._staging_events.pop(stale,None); self._clear_bank_device_views(stale)
        if next_layer is not None:self._prefetch(next_layer)
        if len(self._staging)>2:raise RuntimeError('Memory staging exceeded the fixed two-slot budget')
        self._last_access_layer=layer

    @property
    def staging_layer_ids(self): return tuple(self._staging)
    def clear_active_memory(self):
        self._clear_staging()
        for bank in self.banks.values(): bank.clear_active_memory()
        self._selected_query_chunk=None; self._selected_chunk_ids=()
    def append_to_attention(self,*args,**kwargs):raise RuntimeError('select a layer bank with for_layer()')
