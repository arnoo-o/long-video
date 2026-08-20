"""Sightline training entrypoint; formal training is intentionally not started."""
import argparse
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--data',required=True); p.add_argument('--correspondence-cache',required=True); p.add_argument('--max-steps',type=int,default=2000); p.add_argument('--dry-run',action='store_true'); a=p.parse_args()
 if not a.dry_run: raise SystemExit('Training is intentionally disabled; pass --dry-run for validation.')
 print('sightline config validated; no WAH/PointWorld/ReCal3R/Pi3X initialization')
if __name__=='__main__': main()
