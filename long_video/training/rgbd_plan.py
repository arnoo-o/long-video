"""Soft RGB-D target plans on the real Helios stage grid."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
from ..sightline.rays import anchored_inverse_c2w, latent_camera_indices, token_rays_for_shape


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
    bucket_selected_counts: tuple = ()
    bucket_motion_stats: tuple = ()
    wrong_query_rays: torch.Tensor|None = None
    wrong_key_rays: torch.Tensor|None = None
    separation_px: torch.Tensor|None = None
    spacetime_query_indices: torch.Tensor|None = None
    spacetime_target_indices: torch.Tensor|None = None
    spacetime_target_weights: torch.Tensor|None = None
    spacetime_target_mask: torch.Tensor|None = None
    spacetime_query_weights: torch.Tensor|None = None
    spacetime_bucket: torch.Tensor|None = None
    spacetime_wrong_query_rays: torch.Tensor|None = None
    spacetime_separation_px: torch.Tensor|None = None

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
                                c2w: torch.Tensor|None = None, intrinsics: torch.Tensor|None = None,
                                near_depth: float|None = None):
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
        query_depth=float(row.get('query_depth',float('nan')))
        if not (np.isfinite((qu,qv,ku,kv,query_depth)).all() and query_depth>0 and 0<=qv<480 and 0<=qu<source_width and 0<=kv<480 and 0<=ku<source_width):
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
        bucket=grouped.setdefault((q_index,global_k),{'target':{},'confidence':[],'pixel_count':0,'valid_depth':float(valid_depth_value),'motion':[],'points':[]})
        bucket['pixel_count']+=1; bucket['confidence'].append(confidence); bucket['motion'].append(math.hypot(ku-qu,kv-qv))
        bucket['points'].append((qu,qv,query_depth,confidence))
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
    # Build the query-level spacetime candidates before row sampling.  A
    # query's support is the union of every valid key time; a repeated 3D
    # point at another time remains a valid member of that union.
    spacetime_grouped={}
    for q_index,key_time,target,coverage,confidence,motion in entries:
        value=spacetime_grouped.setdefault(q_index,{'target':{},'pixel_count':0,'valid_depth':1.0,'confidence':[],'motion':[],'points':[]})
        for full_index,target_weight in target.items():
            value['target'][full_index]=value['target'].get(full_index,0.0)+float(target_weight)
        original=grouped[(int(q_index),int(key_time))]
        value['pixel_count']+=int(original.get('pixel_count',0))
        value['valid_depth']=max(value['valid_depth'],float(original.get('valid_depth',1.0)))
        value['confidence'].extend(float(x) for x in original.get('confidence',()))
        value['motion'].extend(float(x) for x in original.get('motion',()))
        value['points'].extend((int(key_time),)+tuple(point) for point in original.get('points',()))
    spacetime_candidates=[]
    for q_index,value in spacetime_grouped.items():
        if not value['target']:
            continue
        coverage=min(1.0,float(value['pixel_count'])/max(float(value['valid_depth']),1.0))
        confidence=float(np.mean(value['confidence'])) if value['confidence'] else 0.0
        motion=float(np.average(value['motion'],weights=value['confidence'])) if value['motion'] else 0.0
        spacetime_candidates.append((q_index,value['target'],coverage,confidence,motion,value['points']))
    spacetime_candidates.sort(key=lambda x:(x[4]/16.0,x[0]))
    spacetime_bucketed=[[] for _ in range(4)]
    for candidate in spacetime_candidates:
        m=candidate[4]/16.0
        spacetime_bucketed[0 if m<.5 else 1 if m<1.5 else 2 if m<3 else 3].append(candidate)
    generator=torch.Generator(device='cpu').manual_seed(int(sampling_seed+7919)&((1<<63)-1))
    spacetime_selected=[]
    for values in spacetime_bucketed:
        order=torch.randperm(len(values),generator=generator).tolist()
        spacetime_selected.extend(values[i] for i in order[:min(32,len(values))])
    spacetime_cap=min(128,len(spacetime_candidates))
    if len(spacetime_selected)<spacetime_cap:
        chosen={id(x) for x in spacetime_selected}
        remaining=[x for x in spacetime_candidates if id(x) not in chosen]
        order=torch.randperm(len(remaining),generator=generator).tolist()
        spacetime_selected.extend(remaining[i] for i in order[:spacetime_cap-len(spacetime_selected)])
    spacetime_selected=spacetime_selected[:spacetime_cap]
    spacetime_selected.sort(key=lambda x:x[0])

    # Motion buckets are defined on the aggregated row, not raw points.
    entries.sort(key=lambda x:(x[5]/16.0,x[0],x[1]))
    bucketed=[[] for _ in range(4)]
    for entry in entries:
        m=entry[5]/16.0
        index=0 if m<.5 else 1 if m<1.5 else 2 if m<3 else 3
        bucketed[index].append(entry)
    generator=torch.Generator(device='cpu').manual_seed(int(sampling_seed)&((1<<63)-1))
    selected=[]
    for values in bucketed:
        order=torch.randperm(len(values),generator=generator).tolist()
        selected.extend(values[i] for i in order[:min(256,len(values))])
    if len(selected)<min(max_rows,len(entries)):
        chosen={id(x) for x in selected}; remaining=[x for x in entries if id(x) not in chosen]
        order=torch.randperm(len(remaining),generator=generator).tolist()
        selected.extend(remaining[i] for i in order[:max_rows-len(selected)])
    selected=selected[:max_rows]
    selected.sort(key=lambda x:(x[0],x[1]))
    bucket_selected_counts=tuple(sum(1 for entry in selected if (0 if entry[5]/16.0<.5 else 1 if entry[5]/16.0<1.5 else 2 if entry[5]/16.0<3 else 3)==bucket_index) for bucket_index in range(4))
    bucket_motion_stats=[]
    for bucket_index in range(4):
        values=[float(entry[5]) for entry in selected if (0 if entry[5]/16.0<.5 else 1 if entry[5]/16.0<1.5 else 2 if entry[5]/16.0<3 else 3)==bucket_index]
        ordered=sorted(values)
        bucket_motion_stats.append({
            'selected_count':len(values),
            'motion_px_mean':float(np.mean(values)) if values else 0.0,
            'motion_px_p50':float(np.quantile(ordered,.5)) if values else 0.0,
            'motion_px_p90':float(np.quantile(ordered,.9)) if values else 0.0,
        })
    # RGB-D prefix captures every spatial K token for each involved key time;
    # target support itself may be small, but the probability normalizer must
    # see the complete legal same-time spatial axis.
    # Keep every key time from the pre-sampling legal set so the spacetime
    # normalizer covers the complete current chunk (9 times x spatial grid).
    # Spacetime normalization is over the complete current chunk, not only
    # times which happened to contribute a sampled target row.
    involved_times=set(range(int(chunk)*8,int(chunk)*8+9))
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
    spacetime_query_indices=torch.as_tensor([entry[0] for entry in spacetime_selected],dtype=torch.long,device=device)
    spacetime_max_support=max((len(entry[1]) for entry in spacetime_selected),default=0)
    spacetime_target_indices=torch.full((len(spacetime_selected),spacetime_max_support),-1,dtype=torch.long,device=device)
    spacetime_target_weights=torch.zeros((len(spacetime_selected),spacetime_max_support),dtype=torch.float32,device=device)
    for row,(_,target,_,_,_,_) in enumerate(spacetime_selected):
        total=max(sum(target.values()),1e-12)
        values=sorted(target.items())
        spacetime_target_indices[row,:len(values)]=torch.as_tensor([sparse_set[index] for index,_ in values],device=device)
        spacetime_target_weights[row,:len(values)]=torch.as_tensor([float(value)/total for _,value in values],device=device)
    spacetime_target_mask=spacetime_target_indices.ge(0)
    spacetime_query_weights=torch.as_tensor([entry[2]*entry[3] for entry in spacetime_selected],dtype=torch.float32,device=device)
    spacetime_motions=torch.as_tensor([entry[4] for entry in spacetime_selected],dtype=torch.float32,device=device)
    spacetime_bucket=torch.where(
        spacetime_motions/16.<.5, torch.zeros_like(spacetime_motions,dtype=torch.long),
        torch.where(spacetime_motions/16.<1.5, torch.ones_like(spacetime_motions,dtype=torch.long),
                    torch.where(spacetime_motions/16.<3, torch.full_like(spacetime_motions,2,dtype=torch.long),
                                torch.full_like(spacetime_motions,3,dtype=torch.long))))
    # Fill each sparse plan tensor with one device-side indexed write instead
    # of launching one CUDA assignment per target/key token.
    target_indices.copy_(torch.as_tensor(target_index_rows,dtype=torch.long,device=device))
    target_values.copy_(torch.as_tensor(target_value_rows,dtype=torch.float32,device=device))
    target_mask.copy_(target_indices.ge(0))
    if legal_row_indices:
        legal_mask[torch.as_tensor(legal_row_indices,dtype=torch.long,device=device),torch.as_tensor(legal_column_indices,dtype=torch.long,device=device)]=True
    wrong_query_rays=wrong_key_rays=None
    separation_values=None
    spacetime_wrong_query_rays=None
    spacetime_separation_values=None
    if c2w is not None and intrinsics is not None:
        camera_indices=latent_camera_indices().tolist()
        poses=c2w if c2w.ndim==4 else c2w.unsqueeze(0)
        Ks=intrinsics if intrinsics.ndim==4 else intrinsics.unsqueeze(0)
        wrong_poses=anchored_inverse_c2w(poses)
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
        spacetime_q_values=[]
        for entry in spacetime_selected:
            identity=identities[int(entry[0])]; temporal=int(identity[1][0])-int(chunk)*8
            spacetime_q_values.append(qgrid[:,temporal*H*W+int(identity[2])*W+int(identity[3])])
        spacetime_wrong_query_rays=torch.stack(spacetime_q_values,1).detach() if spacetime_q_values else None
        separation=[]
        pose_inverse=torch.linalg.inv(poses)
        wrong_pose_inverse=torch.linalg.inv(wrong_poses)
        depth_scale=1.0 if near_depth is None else 1.0/float(near_depth)
        for entry in selected:
            identity=identities[int(entry[0])]; qt=int(identity[1][0])-int(chunk)*8; kt=int(entry[1])-int(chunk)*8
            q_camera=int(camera_indices[qt]); k_camera=int(camera_indices[kt])
            point_values=grouped[(int(entry[0]),int(entry[1]))].get('points',())
            point_separations=[]; point_weights=[]
            for u,v,depth,confidence in point_values:
                pix=torch.tensor([float(u),float(v),1.0],device=poses.device,dtype=poses.dtype).view(1,3,1).expand(poses.shape[0],-1,-1)
                xq=(float(depth)*depth_scale)*torch.linalg.solve(Ks[:,q_camera],pix).squeeze(-1)
                world=(poses[:,q_camera,:3,:3]@xq.unsqueeze(-1)).squeeze(-1)+poses[:,q_camera,:3,3]
                correct=(pose_inverse[:,k_camera,:3,:3]@world.unsqueeze(-1)).squeeze(-1)+pose_inverse[:,k_camera,:3,3]
                wrong_world=(wrong_poses[:,q_camera,:3,:3]@xq.unsqueeze(-1)).squeeze(-1)+wrong_poses[:,q_camera,:3,3]
                wrong=(wrong_pose_inverse[:,k_camera,:3,:3]@wrong_world.unsqueeze(-1)).squeeze(-1)+wrong_pose_inverse[:,k_camera,:3,3]
                uv_correct=(Ks[:,k_camera]@correct.unsqueeze(-1)).squeeze(-1); uv_correct=uv_correct[:,:2]/uv_correct[:,2:].clamp_min(1e-12)
                uv_wrong=(Ks[:,k_camera]@wrong.unsqueeze(-1)).squeeze(-1); uv_wrong=uv_wrong[:,:2]/uv_wrong[:,2:].clamp_min(1e-12)
                distance=torch.linalg.vector_norm(uv_wrong-uv_correct,dim=-1)
                distance=torch.where((correct[:,2]>0)&(wrong[:,2]>0),distance,torch.full_like(distance,float('inf')))
                point_separations.append(distance); point_weights.append(float(confidence))
            if point_separations:
                distances=torch.stack(point_separations,dim=1)
                weights_tensor=torch.as_tensor(point_weights,device=poses.device,dtype=poses.dtype).view(1,-1)
                finite=bool(torch.isfinite(distances).all())
                separation.append((distances*weights_tensor).sum(1)/weights_tensor.sum().clamp_min(1e-12) if finite else torch.full((poses.shape[0],),float('inf'),device=poses.device,dtype=poses.dtype))
            else:
                separation.append(torch.zeros((poses.shape[0],),device=poses.device,dtype=poses.dtype))
        separation_values=torch.stack(separation).detach() if separation else None
        spacetime_separation=[]
        for entry in spacetime_selected:
            q_index=int(entry[0]); identity=identities[q_index]; qt=int(identity[1][0])-int(chunk)*8
            point_separations=[]; point_weights=[]
            for point in entry[5]:
                key_time,u,v,depth,confidence=point
                kt=int(key_time)-int(chunk)*8
                q_camera=int(camera_indices[qt]); k_camera=int(camera_indices[kt])
                pix=torch.tensor([float(u),float(v),1.0],device=poses.device,dtype=poses.dtype).view(1,3,1).expand(poses.shape[0],-1,-1)
                xq=(float(depth)*depth_scale)*torch.linalg.solve(Ks[:,q_camera],pix).squeeze(-1)
                world=(poses[:,q_camera,:3,:3]@xq.unsqueeze(-1)).squeeze(-1)+poses[:,q_camera,:3,3]
                correct=(pose_inverse[:,k_camera,:3,:3]@world.unsqueeze(-1)).squeeze(-1)+pose_inverse[:,k_camera,:3,3]
                wrong_world=(wrong_poses[:,q_camera,:3,:3]@xq.unsqueeze(-1)).squeeze(-1)+wrong_poses[:,q_camera,:3,3]
                wrong=(wrong_pose_inverse[:,k_camera,:3,:3]@wrong_world.unsqueeze(-1)).squeeze(-1)+wrong_pose_inverse[:,k_camera,:3,3]
                uv_correct=(Ks[:,k_camera]@correct.unsqueeze(-1)).squeeze(-1); uv_correct=uv_correct[:,:2]/uv_correct[:,2:].clamp_min(1e-12)
                uv_wrong=(Ks[:,k_camera]@wrong.unsqueeze(-1)).squeeze(-1); uv_wrong=uv_wrong[:,:2]/uv_wrong[:,2:].clamp_min(1e-12)
                distance=torch.linalg.vector_norm(uv_wrong-uv_correct,dim=-1)
                distance=torch.where((correct[:,2]>0)&(wrong[:,2]>0),distance,torch.full_like(distance,float('inf')))
                point_separations.append(distance); point_weights.append(float(confidence))
            if point_separations:
                distances=torch.stack(point_separations,dim=1)
                weights_tensor=torch.as_tensor(point_weights,device=poses.device,dtype=poses.dtype).view(1,-1)
                finite=bool(torch.isfinite(distances).all())
                spacetime_separation.append((distances*weights_tensor).sum(1)/weights_tensor.sum().clamp_min(1e-12) if finite else torch.full((poses.shape[0],),float('inf'),device=poses.device,dtype=poses.dtype))
            else:
                spacetime_separation.append(torch.zeros((poses.shape[0],),device=poses.device,dtype=poses.dtype))
        spacetime_separation_values=torch.stack(spacetime_separation).detach() if spacetime_separation else None
        if separation_values is not None and separation_values.ndim>1: separation_values=separation_values.mean(-1)
        if spacetime_separation_values is not None and spacetime_separation_values.ndim>1: spacetime_separation_values=spacetime_separation_values.mean(-1)
    return RGBDSoftTargetPlan(
        query_indices,torch.as_tensor(sparse,dtype=torch.long,device=device),target_indices,target_values,target_mask,legal_mask,
        torch.as_tensor(query_weights,dtype=torch.float32,device=device),torch.as_tensor(key_times,dtype=torch.long,device=device),
        torch.as_tensor(coverages,dtype=torch.float32,device=device),torch.as_tensor(confidences,dtype=torch.float32,device=device),
        tuple(identities),tuple(map(int,stage_shape)),int(input_count),len(selected),torch.as_tensor(motions,dtype=torch.float32,device=device),torch.as_tensor(motions,dtype=torch.float32,device=device)/16.,torch.as_tensor(buckets,dtype=torch.long,device=device),bucket_selected_counts,tuple(bucket_motion_stats),wrong_query_rays,wrong_key_rays,separation_values.to(device) if separation_values is not None and device is not None else separation_values,
        spacetime_query_indices=spacetime_query_indices,spacetime_target_indices=spacetime_target_indices,
        spacetime_target_weights=spacetime_target_weights,spacetime_target_mask=spacetime_target_mask,
        spacetime_query_weights=spacetime_query_weights,spacetime_bucket=spacetime_bucket,
        spacetime_wrong_query_rays=spacetime_wrong_query_rays,
        spacetime_separation_px=spacetime_separation_values.to(device) if spacetime_separation_values is not None and device is not None else spacetime_separation_values)


__all__=['RGBDSoftTargetPlan','build_rgbd_soft_target_plan']
