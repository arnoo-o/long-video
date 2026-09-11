import pytest
import torch

from long_video.training.rgbd_probability import rgbd_probability_score, rgbd_probability_row_scores_multi, rgbd_log_mass_rows


def _dense_score(native_q, native_k, dq, dk, target_indices, target_weights,
                 target_mask, legal_mask, local_scale, row_weights):
    b, rows, heads, dim = native_q.shape
    q=native_q.detach()+local_scale*dq
    k=native_k.detach()+local_scale*dk
    logits=torch.einsum('brhd,bkhd->brhk',q,k)*dim**-0.5
    masked_logits=logits.masked_fill(~legal_mask[None,:,None,:],-torch.inf)
    logp=logits-torch.logsumexp(masked_logits,-1,keepdim=True)
    target=torch.zeros(rows,k.shape[1],device=q.device,dtype=torch.float32)
    for row in range(rows):
        valid=target_mask[row]&target_indices[row].ge(0)
        target[row].index_add_(0,target_indices[row,valid],target_weights[row,valid].float())
    target=target/target.sum(-1,keepdim=True).clamp_min(1e-12)
    logpbar=torch.logsumexp(logp,dim=2)-torch.log(torch.as_tensor(heads,device=q.device,dtype=torch.float32))
    target_broadcast=target[None]
    scores=(target_broadcast*logpbar.masked_fill(~target_broadcast.gt(0),0.0)).sum(-1)
    return (scores*row_weights[None]).sum()/(row_weights.sum()*b).clamp_min(1e-12)


def test_streaming_probability_forward_and_gradients_match_dense_reference():
    torch.manual_seed(812)
    b,rows,heads,dim,keys=2,4,3,5,13
    native_q=torch.randn(b,rows,heads,dim)
    native_k=torch.randn(b,keys,heads,dim)
    dq=torch.randn_like(native_q,requires_grad=True)
    dk=torch.randn_like(native_k,requires_grad=True)
    target_indices=torch.tensor([[1,2,2,3],[4,5,6,-1],[7,8,-1,-1],[9,10,11,12]])
    target_weights=torch.tensor([[.2,.4,.9,.1],[1.7,.3,.8,0.],[.5,2.,0.,0.],[.1,.2,.3,.4]])
    target_mask=target_indices.ge(0)
    legal=torch.ones(rows,keys,dtype=torch.bool)
    legal[1,0]=False; legal[1,1]=False
    legal[2,0:3]=False
    row_weights=torch.tensor([.7,1.4,.3,2.1])
    dense_q=dq.detach().clone().requires_grad_(True)
    dense_k=dk.detach().clone().requires_grad_(True)
    dense=_dense_score(native_q,native_k,dense_q,dense_k,target_indices,target_weights,target_mask,legal,.73,row_weights)
    streamed=rgbd_probability_score(native_q,native_k,dq,dk,target_indices,target_weights,target_mask,legal,.73,row_weights)
    dense_grad=torch.autograd.grad(.37*dense,(dense_q,dense_k),retain_graph=True)
    (.37*streamed).backward()
    assert torch.allclose(streamed,dense,atol=3e-6,rtol=3e-6)
    assert torch.allclose(dq.grad,dense_grad[0],atol=5e-6,rtol=5e-6)
    assert torch.allclose(dk.grad,dense_grad[1],atol=5e-6,rtol=5e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_streaming_probability_multi_scale_matches_dense_reference_and_gradients():
    torch.manual_seed(1907)
    device=torch.device('cuda')
    b,rows,heads,dim,keys=2,5,3,7,23
    native_q=torch.randn(b,rows,heads,dim,device=device)
    native_k=torch.randn(b,keys,heads,dim,device=device)
    dq=torch.randn_like(native_q,requires_grad=True)
    dk=torch.randn_like(native_k,requires_grad=True)
    indices=torch.tensor([[1,2,2,3,4],[5,6,7,-1,8],[9,10,11,12,-1],[13,14,15,16,17],[18,19,-1,-1,-1]],device=device)
    weights=torch.tensor([[.2,.4,.9,.1,.3],[1.7,.3,.8,0.,.4],[.5,2.,.1,.4,0.],[.1,.2,.3,.4,.5],[.6,.4,0.,0.,0.]],device=device)
    mask=indices.ge(0)
    legal=torch.ones(rows,keys,device=device,dtype=torch.bool)
    legal[1,0:2]=False; legal[2,0:4]=False; legal[4,22]=False
    scales=[0.0,0.17,0.63,1.0]
    multi=rgbd_probability_row_scores_multi(native_q,native_k,dq,dk,indices,weights,mask,legal,scales)
    upstream=torch.tensor([[.37,-.11,.23,.91,-.44],[.19,.73,-.29,.17,.61],[.41,-.27,.58,.33,-.15],[.82,.06,-.47,.29,.38]],device=device)
    (multi*upstream).sum().backward()
    multi_q, multi_k=dq.grad.detach().clone(),dk.grad.detach().clone()
    ref_q=torch.zeros_like(dq); ref_k=torch.zeros_like(dk)
    ref_values=[]
    for scale in scales:
        q=dq.detach().clone().requires_grad_(True); k=dk.detach().clone().requires_grad_(True)
        dense_q=native_q.detach()+scale*q; dense_k=native_k.detach()+scale*k
        logits=torch.einsum('brhd,bkhd->brhk',dense_q,dense_k)*dim**-0.5
        masked=logits.masked_fill(~legal[None,:,None,:],-torch.inf)
        logp=logits-torch.logsumexp(masked,-1,keepdim=True)
        target=torch.zeros(rows,keys,device=device,dtype=torch.float32)
        for row in range(rows):
            valid=mask[row]&indices[row].ge(0)
            target[row].index_add_(0,indices[row,valid],weights[row,valid].float())
        target=target/target.sum(-1,keepdim=True).clamp_min(1e-12)
        logpbar=torch.logsumexp(logp,dim=2)-torch.log(torch.as_tensor(heads,device=device,dtype=torch.float32))
        row_scores=(target[None]*logpbar.masked_fill(~target[None].gt(0),0.0)).sum(-1).mean(0)
        ref_values.append(row_scores.detach())
        (row_scores*upstream[len(ref_values)-1]).sum().backward()
        ref_q.add_(q.grad); ref_k.add_(k.grad)
    ref=torch.stack(ref_values)
    assert torch.allclose(multi,ref,atol=3e-6,rtol=3e-6)
    assert torch.allclose(multi_q,ref_q,atol=2e-5,rtol=2e-5)
    assert torch.allclose(multi_k,ref_k,atol=2e-5,rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_streaming_probability_bf16_matches_dense_reference_on_cuda():
    torch.manual_seed(17)
    device=torch.device('cuda')
    b,rows,heads,dim,keys=2,3,2,8,17
    native_q=torch.randn(b,rows,heads,dim,device=device,dtype=torch.bfloat16)
    native_k=torch.randn(b,keys,heads,dim,device=device,dtype=torch.bfloat16)
    dq=torch.randn_like(native_q,requires_grad=True)
    dk=torch.randn_like(native_k,requires_grad=True)
    indices=torch.tensor([[1,2,3],[4,5,-1],[6,7,8]],device=device)
    weights=torch.tensor([[.2,.5,.3],[.8,.2,0.],[.4,.3,.3]],device=device)
    mask=indices.ge(0); legal=torch.ones(rows,keys,device=device,dtype=torch.bool)
    dense=_dense_score(native_q,native_k,dq.detach().clone().requires_grad_(True),dk.detach().clone().requires_grad_(True),indices,weights,mask,legal,.8,torch.ones(rows,device=device))
    streamed=rgbd_probability_score(native_q,native_k,dq,dk,indices,weights,mask,legal,.8,torch.ones(rows,device=device))
    assert torch.allclose(streamed.float(),dense.float(),atol=2e-2,rtol=2e-2)


def _dense_log_mass(native_q,native_k,dq,dk,indices,weights,mask,legal,local_scale):
    b,rows,heads,dim=native_q.shape
    q=native_q.detach()+local_scale*dq; k=native_k.detach()+local_scale*dk
    logits=torch.einsum('brhd,bkhd->brhk',q,k)*dim**-0.5
    legal_b=legal.unsqueeze(0).unsqueeze(2)
    probs=torch.softmax(logits.masked_fill(~legal_b,-torch.inf),dim=-1)
    target=torch.zeros(b,rows,k.shape[1],device=q.device,dtype=torch.float32)
    safe=indices.clamp_min(0)
    valid=mask&indices.ge(0)&weights.gt(0)
    for batch in range(b):
        for row in range(rows):
            row_legal=legal[row,safe[row]]
            row_valid=valid[row]&row_legal
            target[batch,row].index_add_(0,safe[row,row_valid],weights[row,row_valid].float())
    target=target/target.sum(-1,keepdim=True).clamp_min(1e-12)
    mass=(probs*target.unsqueeze(2)).sum(-1).mean(2)
    return mass.clamp_min(1e-12).log().mean(0)


def test_streaming_log_mass_matches_dense_forward_and_qk_gradients():
    torch.manual_seed(4401)
    b,rows,heads,dim,keys=2,6,3,5,19
    native_q=torch.randn(b,rows,heads,dim)
    native_k=torch.randn(b,keys,heads,dim)
    indices=torch.tensor([[1,2,2,7,8,9],[3,4,5,-1,-1,-1],[6,7,8,9,10,-1],[11,12,-1,-1,-1,-1],[13,14,15,16,-1,-1],[17,18,1,1,-1,-1]])
    weights=torch.tensor([[.2,.4,.9,.1,.3,.2],[1.7,.3,.8,0.,0.,0.],[.5,2.,.1,.4,.2,0.],[.1,.2,0.,0.,0.,0.],[.1,.2,.3,.4,0.,0.],[.6,.4,.2,.1,0.,0.]])
    mask=indices.ge(0)
    legal=torch.ones(rows,keys,dtype=torch.bool)
    legal[1,0:2]=False; legal[2,0:5]=False; legal[4,18]=False
    scale=.73
    dq=torch.randn_like(native_q,requires_grad=True); dk=torch.randn_like(native_k,requires_grad=True)
    ref_q=dq.detach().clone().requires_grad_(True); ref_k=dk.detach().clone().requires_grad_(True)
    dense=_dense_log_mass(native_q,native_k,ref_q,ref_k,indices,weights,mask,legal,scale)
    streamed=rgbd_log_mass_rows(native_q,native_k,dq,dk,indices,weights,mask,legal,scale)
    upstream=.37
    dense_grad=torch.autograd.grad(upstream*dense.sum(),(ref_q,ref_k),retain_graph=True)
    (upstream*streamed.sum()).backward()
    assert torch.allclose(streamed,dense,atol=3e-6,rtol=3e-6)
    assert torch.allclose(dq.grad,dense_grad[0],atol=8e-6,rtol=8e-6)
    assert torch.allclose(dk.grad,dense_grad[1],atol=8e-6,rtol=8e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_streaming_log_mass_bf16_cuda_matches_dense():
    torch.manual_seed(4402)
    device=torch.device('cuda'); b,rows,heads,dim,keys=2,4,2,8,13
    native_q=torch.randn(b,rows,heads,dim,device=device,dtype=torch.bfloat16)
    native_k=torch.randn(b,keys,heads,dim,device=device,dtype=torch.bfloat16)
    indices=torch.tensor([[1,2,2,3],[4,5,6,-1],[7,8,-1,-1],[9,10,11,12]],device=device)
    weights=torch.tensor([[.2,.4,.9,.1],[1.7,.3,.8,0.],[.5,2.,0.,0.],[.1,.2,.3,.4]],device=device)
    mask=indices.ge(0); legal=torch.ones(rows,keys,device=device,dtype=torch.bool); legal[1,0:2]=False; legal[2,0:3]=False
    dq=torch.randn_like(native_q,requires_grad=True); dk=torch.randn_like(native_k,requires_grad=True)
    ref_q=dq.detach().clone().requires_grad_(True); ref_k=dk.detach().clone().requires_grad_(True)
    dense=_dense_log_mass(native_q,native_k,ref_q,ref_k,indices,weights,mask,legal,.8)
    streamed=rgbd_log_mass_rows(native_q,native_k,dq,dk,indices,weights,mask,legal,.8)
    assert torch.allclose(streamed.float(),dense.float(),atol=2e-2,rtol=2e-2)
