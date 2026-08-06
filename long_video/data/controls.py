import numpy as np


def _rotation(axis,angle):
    axis=np.asarray(axis,np.float32); axis/=np.linalg.norm(axis)
    x,y,z=axis; c,s=np.cos(angle),np.sin(angle); q=1-c
    return np.array([[c+x*x*q,x*y*q-z*s,x*z*q+y*s],[y*x*q+z*s,c+y*y*q,y*z*q-x*s],[z*x*q-y*s,z*y*q+x*s,c+z*z*q]],np.float32)


def integrate_controls(initial_c2w, controls, move_speed=1.0, scale=None,
                       max_pitch_degrees=85.0, max_delta_time=0.25):
    """Integrate OpenCV c2w controls with scale-aware translation."""
    pose=np.asarray(initial_c2w,np.float32).copy(); output=[]
    pitch_total=float(-np.arcsin(np.clip(pose[1,2],-1.0,1.0)))
    if pose.shape != (4,4):
        raise ValueError("initial_c2w must have shape [4,4]")
    if scale is not None and getattr(scale, "mode", "relative") != "relative":
        move_speed = float(move_speed) / float(scale.meters_per_world_unit)
    if not np.isfinite(move_speed) or move_speed < 0:
        raise ValueError("move_speed must be finite and non-negative")
    for item in controls:
        dt=float(item.get("delta_time",1/30)); yaw=float(item.get("yaw_delta",0)); requested=float(item.get("pitch_delta",0))
        if not np.isfinite(dt) or dt <= 0 or dt > max_delta_time:
            raise ValueError(f"delta_time must be in (0,{max_delta_time}], got {dt}")
        if not np.isfinite(yaw) or not np.isfinite(requested):
            raise ValueError("yaw_delta and pitch_delta must be finite")
        limit=np.deg2rad(float(max_pitch_degrees))
        new_pitch=np.clip(pitch_total+requested,-limit,limit)
        pitch=float(new_pitch-pitch_total); pitch_total=float(new_pitch)
        pose[:3,:3]=pose[:3,:3]@_rotation([0,1,0],yaw)@_rotation([1,0,0],pitch)
        u,_,vt=np.linalg.svd(pose[:3,:3]); rotation=u@vt
        if np.linalg.det(rotation)<0: u[:,-1]*=-1; rotation=u@vt
        pose[:3,:3]=rotation.astype(np.float32)
        local=np.array([item.get("strafe_right",0)-item.get("strafe_left",0),0,item.get("forward",0)-item.get("backward",0)],np.float32)
        pose[:3,3]+=pose[:3,:3]@(local*move_speed*dt); output.append(pose.copy())
    return np.stack(output) if output else np.empty((0,4,4),np.float32)
