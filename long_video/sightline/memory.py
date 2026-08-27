"""Chunk-atomic K/V memory with a fixed anchor and geometry retrieval."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class MemoryToken:
    chunk_index:int; token_index:int; temporal:int; pooled_y:int; pooled_x:int


@dataclass
class MemoryChunk:
    chunk_id:int
    hidden:torch.Tensor
    rays:torch.Tensor
    temporal:tuple[int,...]
    pooled_y:tuple[int,...]
    pooled_x:tuple[int,...]
    camera_poses:torch.Tensor
    global_latent_ids:tuple[int,...]
    timestamp:int

    def __post_init__(self):
        count=int(self.hidden.shape[1])
        if self.hidden.ndim!=3 or self.rays.ndim!=3 or self.rays.shape[:2]!=self.hidden.shape[:2] or self.rays.shape[-1]!=7:
            raise ValueError('MemoryChunk hidden/rays must be [B,N,D] and [B,N,7]')
        if any(len(values)!=count for values in (self.temporal,self.pooled_y,self.pooled_x)):
            raise ValueError('MemoryChunk metadata must match its token axis')
        if not self.hidden.is_contiguous() or not self.rays.is_contiguous():
            raise ValueError('MemoryChunk tensors must be contiguous')

    @property
    def token_count(self): return int(self.hidden.shape[1])

    @property
    def tokens(self):
        return tuple(MemoryToken(self.chunk_id,index,self.temporal[index],self.pooled_y[index],self.pooled_x[index]) for index in range(self.token_count))


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
    native={int(value) for value in native_history_chunk_ids}
    # Native history has priority: no active Memory chunk may duplicate it.
    past={chunk_id:chunk for chunk_id,chunk in archive.items() if chunk_id<query_chunk and chunk_id not in native}
    selected=[]
    # chunk0 is the permanent anchor, but becomes active only after it leaves
    # Helios' native history window.
    if 0 in past: selected.append(past[0])
    excluded=native|{0,query_chunk}
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
        rr[...,:3]=rr[...,:3]/rr[...,:3].norm(dim=-1,keepdim=True).clamp_min(1e-6)
        rr[...,3:6]=rr[...,3:6]/rr[...,3:6].norm(dim=-1,keepdim=True).clamp_min(1e-6)
        pooled_h,pooled_w=H//self.pool,W//self.pool
        # Discard temporal0 before flattening.  Each payload crosses to CPU as
        # one contiguous tensor; never create or transfer per-token tensors.
        hidden_chunk=hh[:,1:].reshape(hh.shape[0],-1,hh.shape[-1]).contiguous().detach().cpu()
        ray_chunk=rr[:,1:].reshape(rr.shape[0],-1,rr.shape[-1]).contiguous().detach().cpu()
        temporal=tuple(t for t in range(1,T) for _ in range(pooled_h*pooled_w))
        pooled_y=tuple(y for _ in range(1,T) for y in range(pooled_h) for _ in range(pooled_w))
        pooled_x=tuple(x for _ in range(1,T) for _ in range(pooled_h) for x in range(pooled_w))
        global_ids=tuple(int(chunk_index)*8+t for t in temporal)
        chunk=MemoryChunk(int(chunk_index),hidden_chunk,ray_chunk,temporal,pooled_y,pooled_x,camera_poses.detach().cpu(),global_ids,int(chunk_index if timestamp is None else timestamp))
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
                              native_history_chunk_ids=(), device=None, dtype=None):
        """Select and materialize the complete active chunks once per query chunk."""
        query_chunk=int(query_chunk)
        if (self._active_query_chunk == query_chunk and self._active_hidden is not None
                and tuple(native_history_chunk_ids) == getattr(self, '_active_native_history', ())):
            return self._active_tokens
        chunks=self.select_chunks(query_chunk,query_camera_poses,native_history_chunk_ids)
        tokens=tuple(token for chunk in chunks for token in chunk.tokens)
        if len(tokens)>self.budget:
            raise RuntimeError(f'complete active Memory chunks exceed token budget: {len(tokens)}>{self.budget}')
        self.last_active_tokens=list(tokens); self.last_selected_chunk_ids=tuple(chunk.chunk_id for chunk in chunks)
        self._active_query_chunk=query_chunk; self._active_native_history=tuple(int(x) for x in native_history_chunk_ids); self._active_tokens=tokens
        if chunks:
            self._active_hidden=torch.cat([chunk.hidden for chunk in chunks],1).to(device=device,dtype=dtype)
            self._active_rays=torch.cat([chunk.rays for chunk in chunks],1).to(device=device,dtype=dtype)
        else:
            self._active_hidden=self._active_rays=None
        return tokens

    def clear_active_memory(self):
        self._active_query_chunk=None; self._active_native_history=(); self._active_hidden=None; self._active_rays=None; self._active_tokens=()
        self.last_active_tokens=[]; self.last_selected_chunk_ids=()

    def get(self,current_global_start=None,*,query_camera_poses=None,native_history_chunk_ids=(),device=None,dtype=None):
        current_chunk=int(current_global_start//8) if current_global_start is not None else None
        if current_chunk is not None and self._active_query_chunk == current_chunk:
            if self._active_hidden is None: return None,None
            return self._active_hidden.to(device=device,dtype=dtype), self._active_rays.to(device=device,dtype=dtype)
        tokens=self.active_tokens(current_global_start,query_camera_poses=query_camera_poses,native_history_chunk_ids=native_history_chunk_ids)
        if not tokens: return None,None
        chunks=self.select_chunks(current_chunk,query_camera_poses,native_history_chunk_ids)
        hidden=torch.cat([chunk.hidden for chunk in chunks],1); rays=torch.cat([chunk.rays for chunk in chunks],1)
        return hidden.to(device=device,dtype=dtype),rays.to(device=device,dtype=dtype)

    def __len__(self): return sum(chunk.token_count for chunk in self.archive.values())

    def position_metadata(self,tokens,device,current_global_start):
        if not tokens:return None
        global_ids=[token.chunk_index*8+token.temporal for token in tokens]
        frame=torch.tensor([max(1,min(18,19-(current_global_start-value))) for value in global_ids],device=device,dtype=torch.float32)
        offset=(self.pool-1)/2
        y=torch.tensor([token.pooled_y*self.pool+offset for token in tokens],device=device,dtype=torch.float32)
        x=torch.tensor([token.pooled_x*self.pool+offset for token in tokens],device=device,dtype=torch.float32)
        return frame.view(1,-1,1,1),y.view(1,-1,1,1),x.view(1,-1,1,1)

    def memory_rotary_emb(self,tokens,device,current_global_start):
        if self.rope is None: raise RuntimeError('Helios native RoPE module is not bound to memory')
        positions=self.position_metadata(tokens,device,current_global_start)
        if positions is None:return None
        return self.rope.forward_with_positions(*positions,device=device).flatten(2).transpose(1,2)

    def append_native_attention(self,attn,key,value,rotary_emb,rotary_apply,*,current_chunk,current_global_start,timestamp_embedding=None,sightline_projector=None,scale_delta=None,query_camera_poses=None,native_history_chunk_ids=(),**kwargs):
        hidden,rays=self.get(current_global_start,query_camera_poses=query_camera_poses,native_history_chunk_ids=native_history_chunk_ids,device=key.device,dtype=key.dtype)
        tokens=self.last_active_tokens
        if not self.enabled or hidden is None:return key,value,{'memory_tokens':0,'memory_chunk_ids':[]}
        if hidden.shape[0]!=key.shape[0]:
            if hidden.shape[0]==1:hidden=hidden.expand(key.shape[0],-1,-1);rays=rays.expand(key.shape[0],-1,-1)
            else:raise ValueError('memory batch differs from attention batch')
        mem_k=attn.norm_k(attn.to_k(hidden)).unflatten(2,(attn.heads,-1)); mem_v=attn.to_v(hidden).unflatten(2,(attn.heads,-1))
        memory_rotary=self.memory_rotary_emb(tokens,hidden.device,current_global_start).to(mem_k)
        if memory_rotary.shape[1]!=mem_k.shape[1]:raise RuntimeError('memory rotary embedding count mismatch')
        mem_k=rotary_apply(mem_k,memory_rotary)
        if sightline_projector is not None:
            delta=sightline_projector.project(rays.to(hidden),kind='k',training=sightline_projector.training,scale_delta=scale_delta)
            mem_k=mem_k+delta.unflatten(-1,(attn.heads,-1))
        if timestamp_embedding is not None:
            ages=torch.tensor([max(0,current_chunk-token.chunk_index) for token in tokens],device=hidden.device,dtype=torch.long).clamp_max(timestamp_embedding.num_embeddings-1)
            if timestamp_embedding.weight.device!=hidden.device or timestamp_embedding.weight.dtype!=hidden.dtype:raise RuntimeError('memory timestamp embedding device/dtype mismatch')
            mem_k=mem_k+timestamp_embedding(ages).to(mem_k.dtype).unsqueeze(0).unflatten(2,(attn.heads,-1))
        meta={'memory_tokens':mem_k.shape[1],'memory_chunk_count':len(self.last_selected_chunk_ids),'memory_chunk_ids':list(self.last_selected_chunk_ids),'memory_global_ids':[token.chunk_index*8+token.temporal for token in tokens]}
        return torch.cat((key,mem_k),1),torch.cat((value,mem_v),1),meta

    def _memory_rays(self):
        return torch.cat([chunk.rays for chunk in self.archive.values()],1) if self.archive else None


class LayerKVMemoryBank(nn.Module):
    def __init__(self,layers,budget=13312,pool=2,hidden_dim=None,*,tau_pos=1.0,tau_angle=0.78539816339):
        super().__init__(); self.banks={int(layer):LongTermKVMemory(budget,pool,tau_pos=tau_pos,tau_angle=tau_angle) for layer in layers}
        self.timestamp=nn.Embedding(64,hidden_dim) if hidden_dim is not None else None
        if self.timestamp is not None:nn.init.zeros_(self.timestamp.weight)
    def for_layer(self,layer):
        if int(layer) not in self.banks:raise KeyError(f'memory layer {layer} is not selected')
        return self.banks[int(layer)]
    def bind_rope(self,rope):
        for bank in self.banks.values():bank.rope=rope;bank.timestamp=self.timestamp
    def set_enabled(self,enabled):
        for bank in self.banks.values():bank.enabled=bool(enabled)
    def reset(self):
        for bank in self.banks.values():bank.archive.clear();bank.clear_active_memory()
    def prepare_active_memory(self, *, query_chunk:int, query_camera_poses:torch.Tensor, native_history_chunk_ids=(), device=None, dtype=None):
        """Prepare all layer banks once; attention calls only consume this cache."""
        return {layer: bank.prepare_active_memory(query_chunk=query_chunk, query_camera_poses=query_camera_poses,
            native_history_chunk_ids=native_history_chunk_ids, device=device, dtype=dtype) for layer,bank in self.banks.items()}
    def clear_active_memory(self):
        for bank in self.banks.values(): bank.clear_active_memory()
    def append_to_attention(self,*args,**kwargs):raise RuntimeError('select a layer bank with for_layer()')
