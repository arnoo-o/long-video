"""Small smoke probe scaffold; emits machine-readable layer ranking."""
import argparse,json
def main():
 p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--layers',default=''); a=p.parse_args(); layers=[int(x) for x in a.layers.split(',') if x]
 open(a.out,'w').write(json.dumps({'baseline':'helios_source_history','layers':[{'layer':x,'pose_probe':None,'mrr':None,'top1':None,'top5':None,'qk_gap':None} for x in layers]},indent=2))
if __name__=='__main__': main()
