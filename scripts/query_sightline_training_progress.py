"""Read-only Sightline training metrics query."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('metrics',type=Path); a=p.parse_args()
 if not a.metrics.exists(): raise SystemExit('no Sightline metrics found')
 rows=[json.loads(x) for x in a.metrics.read_text().splitlines() if x.strip()]; print(json.dumps(rows[-1] if rows else {},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
