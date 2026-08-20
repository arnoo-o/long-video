"""Geometry-free Sightline inference entrypoint."""
import argparse
def main():
 p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--model',required=True); p.add_argument('--out',required=True); p.add_argument('--chunks',type=int,default=6); p.add_argument('--trajectory'); p.parse_args()
 raise SystemExit('Provide a Helios adapter implementing generate_chunk; no legacy geometry pipeline is permitted.')
if __name__=='__main__': main()
