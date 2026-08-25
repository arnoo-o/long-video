"""End-to-end attention check for the complete alpha-zero baseline switch."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--helios-root',required=True); args=parser.parse_args()
    sys.path.insert(0,args.helios_root)
    import helios.diffusers_version.transformer_helios_diffusers as native
    from long_video.training.sightline import SightlineTrainable,install_lora,configure_alpha_zero_baseline
    from long_video.sightline.memory import LayerKVMemoryBank
    from long_video.sightline.helios_integration import install_sightline_attention
    torch.manual_seed(4)
    attention=native.HeliosAttention(dim=8,heads=2,dim_head=4,is_cross_attention=False,is_amplify_history=False).float()
    hidden=torch.randn(1,5,8); expected=native.HeliosAttnProcessor()(attention,hidden,original_context_length=5)
    class Block(torch.nn.Module):
        def __init__(self,attn): super().__init__(); self.attn1=attn
    class Transformer(torch.nn.Module):
        def __init__(self,attn): super().__init__(); self.transformer_blocks=torch.nn.ModuleList([Block(attn)]); self.rope=None
    class Provider:
        context={'chunk_index':0}
        def __call__(self,states,**kwargs):
            return torch.zeros(states.shape[0],states.shape[1],7),torch.zeros(states.shape[0],kwargs['key_length'],7)
    transformer=Transformer(attention); trainable=SightlineTrainable(8,layers=(0,),heads=2); memory=LayerKVMemoryBank((0,),8,2,hidden_dim=8)
    install_lora(transformer,(0,),rank=8)
    install_sightline_attention(transformer,trainable.conditioner,Provider(),layers=(0,),helios_module=native,memory=memory,memory_layers=(0,))
    configure_alpha_zero_baseline(trainable,memory,transformer)
    actual=attention.processor(attention,hidden,original_context_length=5)
    maximum=float((actual-expected).abs().max().detach())
    if not torch.allclose(actual,expected,rtol=1e-6,atol=1e-7): raise RuntimeError(f'baseline differs from native Helios: max_abs={maximum}')
    print(json.dumps({'passed':True,'max_abs_error':maximum,'qk_residual':False,'memory':False,'lora':False,'value_path':'native'}))
if __name__=='__main__': main()
