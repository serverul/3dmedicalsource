
import trimesh, numpy as np, sys
print('trimesh', trimesh.__version__)
print('engines', trimesh.boolean.engines_available)
from app import make_sample_bone, load_mesh, safe_slice, axis_normal
m=load_mesh(make_sample_bone())
normal=axis_normal('y'); origin=m.bounds.mean(axis=0)
for cap in [False, True]:
 try:
  s=trimesh.intersections.slice_mesh_plane(m, normal, origin, cap=cap)
  print('cap',cap, len(s.faces), s.is_watertight)
 except Exception as e: print('cap',cap,'ERR',type(e).__name__,e)
