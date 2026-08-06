"""Validated YAML configuration loading with dotted CLI overrides."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import yaml

REQUIRED={
    "confidence":("source_prior","token_visible_threshold","token_confidence_threshold","lambda_confidence"),
    "wah":("conditioning_type","warp_history_downsample_mode","rope_alignment"),
    "pi3":("checkpoint","repo_path","device","input_size"),
    "online_memory":("min_transition_frames","keyframe_count","heldout_count",
                     "coverage_threshold","voxel_size"),
    "dit360":("repo_path","python_executable","base_model_path","lora_path",
              "erp_height","erp_width","device"),
}

def _coerce(value):
    try: return json.loads(value)
    except (ValueError,TypeError): return value

def load_yaml(path,overrides=()):
    path=Path(path)
    data=yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data,dict): raise ValueError(f"{path} must contain a mapping")
    result=deepcopy(data)
    for override in overrides:
        if "=" not in override: raise ValueError(f"override must be key=value: {override}")
        dotted,value=override.split("=",1); cursor=result
        parts=dotted.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part],dict): cursor[part]={}
            cursor=cursor[part]
        cursor[parts[-1]]=_coerce(value)
    validate_config(path.stem,result)
    return result

def validate_config(kind,config):
    missing=[key for key in REQUIRED.get(kind,()) if key not in config]
    if missing: raise ValueError(f"{kind} config missing required keys: {missing}")
    if "device" in config and config["device"] in (None,"cuda"):
        raise ValueError(f"{kind}.device must be explicit, e.g. cuda:0 or cpu")
    if kind=="online_memory":
        if config["min_transition_frames"]<config["keyframe_count"]+config["heldout_count"]:
            raise ValueError("transition frames must cover mapping plus held-out frames")
        forbidden=[key for key in config if key.endswith("_m")]
        if forbidden: raise ValueError(f"node-unit config must not use metric suffixes: {forbidden}")
    if kind=="dit360" and (config["erp_height"],config["erp_width"])!=(1024,2048):
        raise ValueError("official DiT360 requires ERP 2048x1024")
    if kind=="wah":
        expected={"conditioning_type":"warp","warp_history_downsample_mode":"short",
                  "rope_alignment":True}
        wrong={k:(config.get(k),v) for k,v in expected.items() if config.get(k)!=v}
        if wrong: raise ValueError(f"WAH spatial-history invariants violated: {wrong}")
    return config
