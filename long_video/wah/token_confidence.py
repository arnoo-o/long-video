import numpy as np
def build_token_confidence(pixel_confidence,pixel_visibility,actual_vae_layout,actual_patch_layout):
    t,h,w=pixel_confidence.shape; vt,ht,wt=actual_vae_layout; pt,ph,pw=actual_patch_layout
    def reduce(a, shape):
        out=np.zeros(shape,np.float32); tt,hh,ww=shape
        for i in range(tt):
            s0=i*a.shape[0]//tt; s1=max(s0+1,(i+1)*a.shape[0]//tt)
            for y in range(hh):
                y0=y*a.shape[1]//hh; y1=max(y0+1,(y+1)*a.shape[1]//hh)
                for x in range(ww):
                    x0=x*a.shape[2]//ww; x1=max(x0+1,(x+1)*a.shape[2]//ww); out[i,y,x]=a[s0:s1,y0:y1,x0:x1].mean()
        return out
    va=reduce(np.asarray(pixel_confidence),(vt,ht,wt)); vv=reduce(np.asarray(pixel_visibility,dtype=np.float32),(vt,ht,wt)); return reduce(va,(pt,ph,pw)), reduce(vv,(pt,ph,pw))
