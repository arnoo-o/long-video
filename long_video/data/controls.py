import numpy as np
def _rotation(axis,angle):
    axis=np.asarray(axis,np.float32); axis/=np.linalg.norm(axis)
    x,y,z=axis; c,s=np.cos(angle),np.sin(angle); q=1-c
    return np.array([[c+x*x*q,x*y*q-z*s,x*z*q+y*s],[y*x*q+z*s,c+y*y*q,y*z*q-x*s],[z*x*q-y*s,z*y*q+x*s,c+z*z*q]],np.float32)
def integrate_controls(initial_c2w,controls,move_speed=1.0):
    pose=np.asarray(initial_c2w,np.float32).copy(); output=[]
    for item in controls:
        dt=float(item.get("delta_time",1/30)); yaw=float(item.get("yaw_delta",0)); pitch=float(item.get("pitch_delta",0))
        pose[:3,:3]=pose[:3,:3]@_rotation([0,1,0],yaw)@_rotation([1,0,0],pitch)
        local=np.array([item.get("strafe_right",0)-item.get("strafe_left",0),0,item.get("backward",0)-item.get("forward",0)],np.float32)
        pose[:3,3]+=pose[:3,:3]@(local*move_speed*dt); output.append(pose.copy())
    return np.stack(output) if output else np.empty((0,4,4),np.float32)
