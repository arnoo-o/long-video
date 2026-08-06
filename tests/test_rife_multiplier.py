import numpy as np

from long_video.oracle_training.dense24 import PracticalRIFE425


def test_rife_gate_uses_midpoint_not_first_eighth(monkeypatch, tmp_path):
    repo=tmp_path/"repo"; checkpoint=tmp_path/"train_log"
    (repo/"model").mkdir(parents=True); checkpoint.mkdir()
    for path in (repo/"model"/"warplayer.py",checkpoint/"RIFE_HDv3.py",checkpoint/"IFNet_HDv3.py",checkpoint/"flownet.pkl"):
        path.write_bytes(b"x")
    adapter=PracticalRIFE425(repo,checkpoint,"python")
    anchors=np.stack([np.zeros((2,3,3),np.uint8),np.full((2,3,3),255,np.uint8)])
    def fake_run(command,**kwargs):
        multiplier=int(command[command.index("--multiplier")+1]); output=command[command.index("--output")+1]
        dense=np.stack([np.rint(anchors[0]*(1-a)+anchors[1]*a).astype(np.uint8) for a in np.linspace(0,1,multiplier+1)])
        np.save(output,dense)
        class Result: returncode=0; stdout=""; stderr=""
        return Result()
    monkeypatch.setattr("subprocess.run",fake_run)
    midpoint=adapter.interpolate(anchors,tmp_path/"two_x",multiplier=2)
    eightfold=adapter.interpolate(anchors,tmp_path/"eight_x",multiplier=8)
    assert len(midpoint)==3 and midpoint[1,0,0,0]==128
    assert len(eightfold)==9 and eightfold[1,0,0,0]==32 and eightfold[4,0,0,0]==128
