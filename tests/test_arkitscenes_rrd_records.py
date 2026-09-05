import json
from pathlib import Path
import numpy as np
import pytest
from long_video.training.rgbd_memory_data import load_rgbd_memory_manifest
from scripts.build_arkitscenes_rrd_records import align_lowres_to_output, match_times, select_indices

def _record(tmp_path: Path):
    root=tmp_path/'record'; (root/'rgb').mkdir(parents=True); (root/'depth').mkdir()
    for i in range(193): (root/'rgb'/f'{i:06d}.png').touch(); (root/'depth'/f'{i:06d}.png').touch()
    target=np.arange(193,dtype=np.float64)/24; indices=np.rint(target*60).astype(np.int32); source=indices/60
    c2w=np.repeat(np.eye(4)[None],193,0); K=np.repeat(np.array([[500,0,416],[0,500,240],[0,0,1.]])[None],193,0)
    for name,value in [('c2w_abs',c2w),('c2w_local',c2w),('intrinsics',K),('timestamps',target),('source_timestamps',source),('source_frame_indices',indices)]: np.save(root/f'{name}.npy',value)
    np.savez(root/'pointcloud.npz',xyz_world=np.zeros((193,3),np.float32),offsets=np.arange(194,dtype=np.int64),source_frame_indices=indices,timestamps=target)
    np.savez(root/'correspondence_cache.npz',query_frame=np.array([32]),key_frame=np.array([0]),query_chunk=np.array([1]),key_chunk=np.array([0]),query_t=np.array([0]),key_t=np.array([0]),query_y=np.array([0]),query_x=np.array([0]),key_y=np.array([0]),key_x=np.array([0]),weight=np.array([1],np.float32))
    timing={key:.001 for key in ('rgb_target_max_error_seconds','depth_rgb_max_error_seconds','pose_rgb_max_error_seconds','intrinsics_rgb_max_error_seconds','depth_intrinsics_rgb_max_error_seconds','confidence_rgb_max_error_seconds')}
    (root/'metadata.json').write_text(json.dumps({'pose_source':'mebx_stream_4_vision_transform','orientation_source':'measured_gravity','timing_validation':timing}))
    row={'record_id':'arkitscenes__1__000000','dataset':'arkitscenes','scene_id':'1','sequence_id':'1','rgb_dir':'record/rgb','depth_dir':'record/depth','c2w_abs':'record/c2w_abs.npy','c2w_local':'record/c2w_local.npy','intrinsics':'record/intrinsics.npy','timestamps':'record/timestamps.npy','source_timestamps':'record/source_timestamps.npy','source_frame_indices':'record/source_frame_indices.npy','pointcloud':'record/pointcloud.npz','metadata':'record/metadata.json','correspondence_cache':'record/correspondence_cache.npz','frame_count':193,'chunk_count':6,'fps':24,'height':480,'width':832,'near_depth':1.0,'near_depth_unit':'m','near_depth_method':'median(frame_valid_depth_q25_m)','memory_eligible':True}
    return row,root

def test_real_60hz_mapping_is_unique_and_within_10ms():
    source=np.arange(600,dtype=np.float64)/60; ids=select_indices(source,1)
    assert ids is not None and len(np.unique(ids))==193 and np.all(np.diff(ids)>0)
    assert match_times(source,source[ids]) is not None

def test_lowres_depth_alignment_respects_principal_point_offset():
    depth=np.zeros((192,256),np.uint16); depth[80,100]=1234
    low=np.array([[200,0,96],[0,200,128],[0,0,1]],np.float64)
    out=np.array([[400,0,426],[0,400,240],[0,0,1]],np.float64)
    aligned=align_lowres_to_output(depth,low,out)
    assert aligned[144,434] == 1234

def test_strict_arkitscenes_rrd_validator(tmp_path):
    row,_=_record(tmp_path); manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps({'records':[row]}))
    assert load_rgbd_memory_manifest(manifest,expected_count=1)[0].chunk_count==6

def test_strict_arkitscenes_rrd_validator_accepts_parent_derived_second_unit(tmp_path):
    row,root=_record(tmp_path)
    source=np.load(root/'source_timestamps.npy')+123.0; target=np.load(root/'timestamps.npy')
    source[96]=123.0+target[96]-.008; source[97]=123.0+target[97]+.008; np.save(root/'source_timestamps.npy',source)
    row.update(record_id='arkitscenes__1__000000__frames_096_192',parent_record_id='arkitscenes__1__000000',source_frame_start=96,frame_count=97,chunk_count=3)
    manifest=tmp_path/'unit_manifest.json'; manifest.write_text(json.dumps({'records':[row]}))
    record=load_rgbd_memory_manifest(manifest,expected_count=1)[0]
    assert record.source_frame_start==96 and record.load_timestamps()[0]==4.0

def test_arkitscenes_rrd_validator_rejects_bad_provenance(tmp_path):
    row,root=_record(tmp_path); metadata=json.loads((root/'metadata.json').read_text()); metadata['pose_source']='lowres_traj_slerp'; (root/'metadata.json').write_text(json.dumps(metadata))
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps({'records':[row]}))
    with pytest.raises(ValueError,match='provenance'): load_rgbd_memory_manifest(manifest)
