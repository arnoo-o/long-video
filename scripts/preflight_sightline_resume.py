#!/usr/bin/env python3
"""Fail-fast audit for checkpoint-599, mixed manifests, and the P2→P3 boundary."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from long_video.config import load_sightline_config
from long_video.training.sightline import curriculum_phase
from long_video.training.sightline_checkpoint import SCHEMA, SEMANTICS, config_fingerprint


def records(path: Path) -> list[dict]:
    value=json.loads(path.read_text(encoding='utf-8')).get('records')
    if not isinstance(value,list): raise ValueError(f'invalid manifest: {path}')
    return value


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--helios-source',type=Path,required=True)
    parser.add_argument('--unit-manifest',type=Path,required=True)
    parser.add_argument('--p3-manifest',type=Path,required=True)
    args=parser.parse_args()
    cfg=load_sightline_config(args.config); config=asdict(cfg)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    checks={
        'completed_step':int(payload.get('step',-1)),
        'next_step':int(payload.get('step',-1))+1,
        'semantics':payload.get('sightline_training_semantics_version'),
        'schema':payload.get('sightline_checkpoint_schema_version'),
        'config_fingerprint_match':payload.get('config_fingerprint')==config_fingerprint(config),
        'helios_fingerprint_match':payload.get('helios_fingerprint')==hashlib.sha256(args.helios_source.read_bytes()).hexdigest(),
        'optimizer':payload.get('optimizer') is not None,
        'scheduler':payload.get('scheduler') is not None,
        'rng':payload.get('rng_states') is not None or all(key in payload for key in ('rng_torch','rng_python','rng_numpy')),
    }
    if checks['completed_step']!=599 or checks['next_step']!=600 or checks['semantics']!=SEMANTICS or checks['schema']!=SCHEMA or not all(checks[key] for key in ('config_fingerprint_match','helios_fingerprint_match','optimizer','scheduler','rng')):
        raise RuntimeError(f'checkpoint-599 resume preflight failed: {checks}')
    phases={step:curriculum_phase(step,p1_steps=cfg.p1_steps,p2_steps=cfg.p2_steps,p3_steps=cfg.p3_steps) for step in (599,600,999,1000,1499,1500,1799,1800,2099,2100,2299,2300,2499)}
    if phases[599]['name']!='P2' or phases[600]['name']!='P2' or phases[999]['name']!='P2' or phases[1000]!={'name':'P3','max_chunks':2,'lora':True,'correspondence':True,'memory':True} or phases[2499]['max_chunks']!=6:
        raise RuntimeError(f'curriculum boundary mismatch: {phases}')
    units=records(args.unit_manifest);p3=records(args.p3_manifest)
    if any(int(row['chunk_count'])!=3 or int(row['frame_count'])!=97 for row in units):
        raise RuntimeError('P1/P2 manifest contains a non-3-chunk unit')
    if any(int(row['chunk_count']) not in (3,6) or int(row['frame_count'])!=1+32*int(row['chunk_count']) for row in p3):
        raise RuntimeError('P3 manifest contains invalid mixed geometry')
    if not any(int(row['chunk_count'])==6 for row in p3):
        raise RuntimeError('P3 has no legal 4–6 chunk trajectories')
    print(json.dumps({'checkpoint':checks,'unit_records':len(units),'p3_records':len(p3),'phases':phases},indent=2))


if __name__=='__main__': main()
