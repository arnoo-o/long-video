"""Bounded source + 16/2/1 latent history with explicit causal indices."""
class HistoryManager:
    def __init__(self, sizes=(16,2,1), chunk_length=33, stride=32):
        if tuple(sizes)!=(16,2,1): raise ValueError("history sizes must be (16,2,1)")
        self.sizes=tuple(sizes); self.chunk_length=chunk_length; self.stride=stride; self._frames={}; self._source=None; self.chunk_index=0
    def set_source(self, latent):
        if self._source is None: self._source=latent
        elif self._source is not latent and getattr(self._source,'shape',None)!=getattr(latent,'shape',None): raise RuntimeError("source already fixed")
    def append_chunk(self, frames):
        if len(frames)!=self.chunk_length: raise ValueError("chunk must contain 33 frames")
        start=self.chunk_index*self.stride
        for i,x in enumerate(frames):
            idx=start+i
            if idx in self._frames: raise RuntimeError("history frame submitted twice")
            self._frames[idx]=x
        self.chunk_index+=1
    def slots(self):
        if self._source is None: raise RuntimeError("source latent is not set")
        keys=sorted(self._frames)
        long=keys[-16:]; mid=keys[-18:-16]; short=keys[-19:-18]
        return [self._source]+[self._frames[k] for k in long+mid+short]
    def indices(self):
        keys=sorted(self._frames); return {'source':0,'long':keys[-16:],'mid':keys[-18:-16],'short':keys[-19:-18]}
