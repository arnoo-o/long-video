"""Sparse offline correspondence helpers (no dense N×N matrices)."""
import torch
def mutual_nearest(a,b,chunk_size=4096):
    if a.ndim!=2 or b.ndim!=2: raise ValueError('nearest-neighbour inputs must be [N,D]')
    ab=[]
    for start in range(0,a.shape[0],chunk_size): ab.append(torch.cdist(a[start:start+chunk_size],b).argmin(1))
    ab=torch.cat(ab); ba=[]
    for start in range(0,b.shape[0],chunk_size): ba.append(torch.cdist(b[start:start+chunk_size],a).argmin(1))
    ba=torch.cat(ba); q=torch.arange(a.shape[0],device=a.device); keep=ba[ab]==q
    return torch.stack((q[keep],ab[keep]),-1)
def cycle_consistent(pairs_ab,pairs_bc,pairs_ca=None):
    """Keep A->B pairs that close a B->C->A cycle (or legacy B->A check)."""
    if pairs_ab.numel()==0:return pairs_ab
    m_bc={int(k):int(v) for k,v in pairs_bc.tolist()}
    if pairs_ca is None:
        keep=[m_bc.get(int(b))==int(a) for a,b in pairs_ab.tolist()]
    else:
        m_ca={int(k):int(v) for k,v in pairs_ca.tolist()}
        keep=[(int(c:=m_bc.get(int(b),-1))>=0 and m_ca.get(c)==int(a)) for a,b in pairs_ab.tolist()]
    return pairs_ab[torch.tensor(keep,device=pairs_ab.device)]
def correspondence_loss(logits, positive, weight=None):
    loss=torch.nn.functional.cross_entropy(logits,positive,reduction='none'); return (loss if weight is None else loss*weight).mean()

def zbuffer_visible(points_world, c2w, intrinsics, height, width, eps=1e-4):
    """Return a visibility mask using a deterministic nearest-depth z-buffer."""
    import torch
    if points_world.ndim != 2 or points_world.shape[-1] != 3: raise ValueError("points_world must be [N,3]")
    w=torch.cat((points_world,torch.ones_like(points_world[:,:1])),-1)
    cam=(torch.linalg.inv(c2w)@w.t()).t(); z=cam[:,2]; uv=(intrinsics@cam[:,:3].t()).t(); uv=uv[:,:2]/uv[:,2:3].clamp_min(eps)
    px=torch.floor(uv[:,0]).long(); py=torch.floor(uv[:,1]).long(); inside=(z>0)&(px>=0)&(px<width)&(py>=0)&(py<height)
    linear=py*width+px; safe=linear.clamp(0,height*width-1); depth=torch.full((height*width,),float('inf'),device=z.device,dtype=z.dtype)
    depth.scatter_reduce_(0,safe[inside],z[inside],reduce='amin',include_self=True)
    return inside & (z <= depth[safe]+eps)

def sparse_rows(query_xyz, key_xyz, *, query_chunk=0, key_chunk=0, threshold=float('inf')):
    """Mutual NN rows with distances as weights; sparse by construction."""
    import torch
    pairs=mutual_nearest(query_xyz,key_xyz)
    if not len(pairs): return []
    dist=(query_xyz[pairs[:,0]]-key_xyz[pairs[:,1]]).norm(dim=-1)
    return [{'query_token_index':int(q),'positive_key_index':int(k),'weight':float(torch.exp(-d).item()),'distance':float(d),'query_chunk':query_chunk,'key_chunk':key_chunk} for (q,k),d in zip(pairs.tolist(),dist.tolist()) if float(d)<=threshold]
