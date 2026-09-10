import pytest
import torch

from long_video.training.rgbd_probability import rgbd_probability_score


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
