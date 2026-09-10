"""Soft RGB-D target plans on the real Helios stage grid."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
from ..sightline.rays import token_rays_for_shape


@dataclass(frozen=True)
class RGBDSoftTargetPlan:
    query_indices: torch.Tensor
    sparse_key_indices: torch.Tensor
    target_indices: torch.Tensor
    target_weights: torch.Tensor
    target_mask: torch.Tensor
    legal_mask: torch.Tensor
    query_weights: torch.Tensor
    query_key_times: torch.Tensor
    coverage: torch.Tensor
    mean_confidence: torch.Tensor
    identities: tuple
    stage_shape: tuple
    mapping_input_count: int
    mapping_output_count: int
    motion_px: torch.Tensor
    motion_stage2: torch.Tensor
    bucket: torch.Tensor
    wrong_query_rays: torch.Tensor|None = None
    wrong_key_rays: torch.Tensor|None = None
    separation_px: torch.Tensor|None = None

    @property
    def weights(self):
        return self.query_weights

    @property
    def positive_mask(self):
        return self.target_mask

    @property
    def negative_pair_count(self):
        # Kept as a log-schema compatibility alias.  Soft targets have no
        # formal negatives; their count is the number of selected rows.
        return int(self.mapping_output_count)

    @property
    def negative_key_t_match(self):
        return True


def _column(rows, name, fallback=None):
    if hasattr(rows, 'column'):
        try:
            return np.asarray(rows.column(name))
        except (KeyError, ValueError):
            pass
    if isinstance(rows, dict):
        if name in rows:
            return np.asarray(rows[name])
        if fallback is not None and fallback in rows:
            return np.asarray(rows[fallback])
    values=[]
    for row in rows:
        if name in row:
            values.append(row[name])
        elif fallback is not None and fallback in row:
            values.append(row[fallback])
        else:
            raise KeyError(name)
    return np.asarray(values)


def _row_value(rows, name, index, fallback=None, default=None):
    try:
        value=_column(rows,name,fallback)
        return value[index]
    except (KeyError, IndexError):
        return default


def _pixel(row, modern, legacy, scale, default=-1.0):
    if not isinstance(row, dict):
        return float(default)
    if modern in row:
        return float(row[modern])
    if legacy in row:
        return (float(row[legacy]) + 0.5) * scale - 0.5
    return float(default)


def _as_rows(rows):
    if hasattr(rows, 'column'):
        n=len(rows)
        names=('query_chunk','query_t','query_y','query_x','key_chunk','key_t','key_y','key_x','weight','query_frame','valid_depth_count')
        # CorrespondenceSlice.column() performs an indexed NumPy gather.  The
        # old implementation called it once per field *per row*, which made
        # plan construction CPU-bound for large caches.  Materialize each
        # available column once, then only perform cheap scalar indexing in
        # the row loop.  This is representation-only: values and ordering are
        # unchanged.
        columns={}
        for name in names:
            try:
                columns[name]=rows.column(name)
            except (KeyError,ValueError):
                pass
        optional=('query_u','query_v','query_depth','key_u_cont','key_v_cont','confidence',
                  'query_valid_depth_count','query_valid_depth_count_stage0',
                  'query_valid_depth_count_stage1','query_valid_depth_count_stage2')
        for name in optional:
            try:
                columns[name]=rows.column(name)
            except (KeyError,ValueError):
                pass
        result=[]
        for i in range(n):
            item={}
            for name,values in columns.items():
                item[name]=values[i]
            result.append(item)
        return result
    return list(rows)


def build_rgbd_soft_target_plan(rows, identities, stage_shape, *, chunk: int,
                                max_rows: int = 1024, sampling_seed: int = 0,
                                device=None, source_height: int = 512,
                                source_width: int = 832,
                                legacy_token_shape: tuple[int,int,int] | None = (9,32,52),
                                c2w: torch.Tensor|None = None, intrinsics: torch.Tensor|None = None):
    """Build a target distribution for one real ``(T,H,W)`` Helios grid.

    The plan owns a sparse K axis containing all spatial K tokens for the
    involved key times.  Target splats are aggregated by query token and key
    time; duplicated pixel votes therefore change confidence, not the number
    of times a row is sampled.
    """
    T,H,W=map(int,stage_shape)
    if T<1 or H<1 or W<1: raise ValueError('stage_shape must be positive')
    rows=_as_rows(rows)
    legacy_h=legacy_token_shape[1] if legacy_token_shape is not None else H
    legacy_w=legacy_token_shape[2] if legacy_token_shape is not None else W
    full_by_identity={}
    current_indices_by_time={}
    for index,identity in enumerate(identities):
        if len(identity)<4: continue
        kind, times, y, x = identity[:4]
        if kind == 'current' and len(times)==1:
            global_time=int(times[0])
            full_index=int(index)
            full_by_identity[(global_time,int(y),int(x))]=full_index
            current_indices_by_time.setdefault(global_time,[]).append(full_index)
    grouped={}
    input_count=0
    motions={}
    for row in rows:
        qchunk=int(row.get('query_chunk',chunk)); kchunk=int(row.get('key_chunk',chunk))
        qt=int(row.get('query_t',row.get('query_latent_temporal',-1))); kt=int(row.get('key_t',row.get('key_latent_temporal',-1)))
        if qchunk!=int(chunk) or kchunk!=int(chunk) or not (0<=qt<T and 0<=kt<T) or qt==kt:
            continue
        qv=_pixel(row,'query_v','query_y',source_height/legacy_h)
        qu=_pixel(row,'query_u','query_x',source_width/legacy_w)
        kv=float(row.get('key_v_cont',_pixel(row,'key_v','key_y',source_height/legacy_h)))
        ku=float(row.get('key_u_cont',_pixel(row,'key_u','key_x',source_width/legacy_w)))
        if not (np.isfinite((qu,qv,ku,kv)).all() and 0<=qv<480 and 0<=qu<source_width and 0<=kv<480 and 0<=ku<source_width):
            continue
        qx=int(math.floor((qu+0.5)*W/source_width)); qy=int(math.floor((qv+0.5)*H/source_height))
        if not (0<=qx<W and 0<=qy<H): continue
        global_q=int(chunk)*8+qt; global_k=int(chunk)*8+kt
        q_index=full_by_identity.get((global_q,qy,qx))
        if q_index is None: continue
        confidence=float(row.get('confidence',row.get('weight',1.0)))
        if not np.isfinite(confidence) or confidence<=0: continue
        input_count+=1
        key_y=(kv+0.5)*H/source_height-0.5; key_x=(ku+0.5)*W/source_width-0.5
        y0=int(math.floor(key_y)); x0=int(math.floor(key_x)); dy=key_y-y0; dx=key_x-x0
        splats=((y0,x0,(1-dy)*(1-dx)),(y0,x0+1,(1-dy)*dx),(y0+1,x0,dy*(1-dx)),(y0+1,x0+1,dy*dx))
        stage_field={8:'query_valid_depth_count_stage0',16:'query_valid_depth_count_stage1',32:'query_valid_depth_count_stage2'}.get(H)
        valid_depth_value=row.get(stage_field,row.get('query_valid_depth_count',row.get('valid_depth_count',1.0))) if stage_field else row.get('query_valid_depth_count',row.get('valid_depth_count',1.0))
        bucket=grouped.setdefault((q_index,global_k),{'target':{},'confidence':[],'pixel_count':0,'valid_depth':float(valid_depth_value),'motion':[]})
        bucket['pixel_count']+=1; bucket['confidence'].append(confidence); bucket['motion'].append(math.hypot(ku-qu,kv-qv))
        for yy,xx,weight in splats:
            if weight<=0 or not (0<=yy<H and 0<=xx<W): continue
            key_index=full_by_identity.get((global_k,yy,xx))
            if key_index is not None:
                bucket['target'][key_index]=bucket['target'].get(key_index,0.0)+confidence*float(weight)
    if not grouped:
        return None
    entries=[]
    for (q_index,key_time),value in grouped.items():
        if not value['target']: continue
        valid_depth=max(float(value['valid_depth']),1.0)
        coverage=min(1.0,float(value['pixel_count'])/valid_depth)
        mean_conf=float(np.mean(value['confidence']))
        motion=float(np.average(value['motion'],weights=value['confidence'])) if value['motion'] else 0.0
        entries.append((q_index,key_time,value['target'],coverage,mean_conf,motion))
    if not entries: return None
    # Motion buckets are defined on the aggregated row, not raw points.
    entries.sort(key=lambda x:(x[5]/16.0,x[0],x[1]))
    bucketed=[[] for _ in range(4)]
    for entry in entries:
        m=entry[5]/16.0
        index=0 if m<.5 else 1 if m<1.5 else 2 if m<3 else 3
        bucketed[index].append(entry)
    selected=[]
    for values in bucketed:
        selected.extend(values[:256])
    if len(selected)<min(max_rows,len(entries)):
        chosen={id(x) for x in selected}; remaining=[x for x in entries if id(x) not in chosen]
        generator=torch.Generator(device='cpu').manual_seed(int(sampling_seed)&((1<<63)-1))
        order=torch.randperm(len(remaining),generator=generator).tolist()
        selected.extend(remaining[i] for i in order[:max_rows-len(selected)])
    selected=selected[:max_rows]
    selected.sort(key=lambda x:(x[0],x[1]))
    # RGB-D prefix captures every spatial K token for each involved key time;
    # target support itself may be small, but the probability normalizer must
    # see the complete legal same-time spatial axis.
    involved_times={int(entry[1]) for entry in selected}
    sparse=sorted({key for entry in selected for key in entry[2]} | {
        int(index) for index,identity in enumerate(identities)
        if len(identity)>=4 and identity[0]=='current' and int(identity[1][0]) in involved_times
    })
    sparse_set={key:i for i,key in enumerate(sparse)}
    max_support=max(len(entry[2]) for entry in selected)
    target_indices=torch.full((len(selected),max_support),-1,dtype=torch.long,device=device)
    target_values=torch.zeros((len(selected),max_support),dtype=torch.float32,device=device)
    target_mask=torch.zeros_like(target_values,dtype=torch.bool)
    legal_mask=torch.zeros((len(selected),len(sparse)),dtype=torch.bool,device=device)
    query_indices=torch.as_tensor([entry[0] for entry in selected],dtype=torch.long,device=device)
    target_index_rows=[]; target_value_rows=[]
    legal_by_time={}
    for key_time in involved_times:
        legal_by_time[key_time]=[sparse_set[full_index] for full_index in current_indices_by_time.get(key_time,()) if full_index in sparse_set]
    legal_row_indices=[]; legal_column_indices=[]
    query_weights=[]; key_times=[]; coverages=[]; confidences=[]; motions=[]; buckets=[]
    for r,entry in enumerate(selected):
        _,key_time,target,coverage,confidence,motion=entry
        total=sum(target.values())
        index_row=[]; value_row=[]
        for p,(full_index,value) in enumerate(sorted(target.items())):
            index_row.append(sparse_set[full_index]); value_row.append(float(value)/max(total,1e-12))
        target_index_rows.append(index_row+[-1]*(max_support-len(index_row)))
        target_value_rows.append(value_row+[0.0]*(max_support-len(value_row)))
        target_times=legal_by_time.get(key_time,())
        legal_row_indices.extend([r]*len(target_times)); legal_column_indices.extend(target_times)
        query_weights.append(coverage*confidence); key_times.append(key_time); coverages.append(coverage); confidences.append(confidence); motions.append(motion)
        m=motion/16.; buckets.append(0 if m<.5 else 1 if m<1.5 else 2 if m<3 else 3)
    # Fill each sparse plan tensor with one device-side indexed write instead
    # of launching one CUDA assignment per target/key token.
    target_indices.copy_(torch.as_tensor(target_index_rows,dtype=torch.long,device=device))
    target_values.copy_(torch.as_tensor(target_value_rows,dtype=torch.float32,device=device))
    target_mask.copy_(target_indices.ge(0))
    if legal_row_indices:
        legal_mask[torch.as_tensor(legal_row_indices,dtype=torch.long,device=device),torch.as_tensor(legal_column_indices,dtype=torch.long,device=device)]=True
    wrong_query_rays=wrong_key_rays=None
    separation_values=None
    if c2w is not None and intrinsics is not None:
        poses=c2w if c2w.ndim==4 else c2w.unsqueeze(0)
        Ks=intrinsics if intrinsics.ndim==4 else intrinsics.unsqueeze(0)
        delta=torch.linalg.inv(poses[:,:1])@poses
        wrong_poses=poses[:,:1]@torch.linalg.inv(delta)
        qgrid=token_rays_for_shape(wrong_poses,Ks,(poses.shape[0],T,H,W,1),source_height=source_height,source_width=source_width,kind='q').reshape(poses.shape[0],-1,7)
        kgrid=token_rays_for_shape(wrong_poses,Ks,(poses.shape[0],T,H,W,1),source_height=source_height,source_width=source_width,kind='k').reshape(poses.shape[0],-1,7)
        q_values=[]
        for entry in selected:
            identity=identities[int(entry[0])]; temporal=int(identity[1][0])-int(chunk)*8
            q_values.append(qgrid[:,temporal*H*W+int(identity[2])*W+int(identity[3])])
        k_values=[]
        for full_index in sparse:
            identity=identities[int(full_index)]; temporal=int(identity[1][0])-int(chunk)*8
            k_values.append(kgrid[:,temporal*H*W+int(identity[2])*W+int(identity[3])])
        wrong_query_rays=torch.stack(q_values,1).detach() if q_values else None
        wrong_key_rays=torch.stack(k_values,1).detach() if k_values else None
        separation=[]
        for entry in selected:
            identity=identities[int(entry[0])]; qt=int(identity[1][0])-int(chunk)*8; kt=int(entry[1])-int(chunk)*8
            u=(float(identity[3])+0.5)*source_width/W-0.5; v=(float(identity[2])+0.5)*source_height/H-0.5
            pix=torch.tensor([u,v,1.0],device=poses.device,dtype=poses.dtype).view(1,3,1).expand(poses.shape[0],-1,-1)
            xq=torch.linalg.solve(Ks[:,qt*4],pix).squeeze(-1)
            world=(poses[:,qt*4,:3,:3]@xq.unsqueeze(-1)).squeeze(-1)+poses[:,qt*4,:3,3]
            correct=(torch.linalg.inv(poses[:,kt*4])[:,:3,:3]@world.unsqueeze(-1)).squeeze(-1)+torch.linalg.inv(poses[:,kt*4])[:,:3,3]
            wrong=(torch.linalg.inv(wrong_poses[:,kt*4])[:,:3,:3]@((wrong_poses[:,qt*4,:3,:3]@xq.unsqueeze(-1)).squeeze(-1)+wrong_poses[:,qt*4,:3,3]).unsqueeze(-1)).squeeze(-1)+torch.linalg.inv(wrong_poses[:,kt*4])[:,:3,3]
            uv_correct=(Ks[:,kt*4]@correct.unsqueeze(-1)).squeeze(-1); uv_correct=uv_correct[:,:2]/uv_correct[:,2:].clamp_min(1e-12)
            uv_wrong=(Ks[:,kt*4]@wrong.unsqueeze(-1)).squeeze(-1); valid=wrong[:,2]>0
            uv_wrong=uv_wrong[:,:2]/uv_wrong[:,2:].clamp_min(1e-12)
            sep=torch.where(valid,torch.linalg.vector_norm(uv_wrong-uv_correct,dim=-1),torch.full_like(valid.float(),float('inf')))
            separation.append(sep.mean())
        separation_values=torch.stack(separation).detach() if separation else None
    return RGBDSoftTargetPlan(
        query_indices,torch.as_tensor(sparse,dtype=torch.long,device=device),target_indices,target_values,target_mask,legal_mask,
        torch.as_tensor(query_weights,dtype=torch.float32,device=device),torch.as_tensor(key_times,dtype=torch.long,device=device),
        torch.as_tensor(coverages,dtype=torch.float32,device=device),torch.as_tensor(confidences,dtype=torch.float32,device=device),
        tuple(identities),tuple(map(int,stage_shape)),int(input_count),len(selected),torch.as_tensor(motions,dtype=torch.float32,device=device),torch.as_tensor(motions,dtype=torch.float32,device=device)/16.,torch.as_tensor(buckets,dtype=torch.long,device=device),wrong_query_rays,wrong_key_rays,separation_values.to(device) if separation_values is not None and device is not None else separation_values)


__all__=['RGBDSoftTargetPlan','build_rgbd_soft_target_plan']
