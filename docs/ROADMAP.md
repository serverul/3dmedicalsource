# Roadmap

## Product direction

BonePlannerCAD is not a generic DICOM viewer. Horos and VPOP already cover much of 2D/viewer planning. The target is:

```txt
3D CT-based patient-specific surgical CAD for bones
```

Core value:

- extract accurate bone model from CT
- virtual osteotomy
- fragment repositioning
- patient-specific cutting/drill guides
- patient-specific plates/implants
- export STL/STEP-like manufacturing artifacts

## Phase 1 — Segmentation Engine v2 (next)

Goal: reliable bone STL from real DICOM CT.

Tasks:

- [ ] Add binary morphology controls:
  - closing
  - opening
  - dilation/erosion
  - fill holes
- [ ] Remove small islands by voxel count / physical volume.
- [ ] ROI crop:
  - automatic bounding box around thresholded bone
  - manual crop controls later
- [ ] Axis alignment:
  - PCA on bone voxels/mesh
  - align long axis to chosen CAD axis
- [ ] Mesh cleanup:
  - smoothing
  - decimation
  - hole fill
  - normals repair
  - manifold/watertight report
- [ ] Add segmentation presets:
  - skull/maxillofacial
  - long bone
  - spine
  - high-density cortical
  - metal artifact cutoff
- [ ] Export segmentation parameters JSON with every STL.

## Phase 2 — DICOM/segmentation preview

Goal: users must see/validate segmentation before CAD.

Tasks:

- [ ] Add axial slice preview in browser.
- [ ] HU window/level controls.
- [ ] Overlay threshold mask on slice.
- [ ] Scroll through slices.
- [ ] Show selected series metadata.
- [ ] Show voxel spacing and physical dimensions.
- [ ] Add warnings for low slice count / anisotropic spacing / extreme HU values.

## Phase 3 — Osteotomy planning

Goal: more precise virtual surgery.

Tasks:

- [ ] Interactive plane widget instead of axis-only plane.
- [ ] Plane from 3 points / landmarks.
- [ ] Multiple osteotomy planes.
- [ ] Store named fragments.
- [ ] Transform gizmo for fragment movement.
- [ ] Before/after measurement tools.
- [ ] Export per-fragment transforms and STL.

## Phase 4 — Conformal surgical guides

Goal: patient-specific guides that fit the bone.

Tasks:

- [ ] Interactive contact patch selection on mesh.
- [ ] Generate negative/contact surface from selected patch.
- [ ] Clearance offset control.
- [ ] Guide body solidification.
- [ ] Saw slot from osteotomy plane/instrument library.
- [ ] Drill/pin sleeves.
- [ ] Cooling/irrigation channel option.
- [ ] Validate thickness and watertightness.

## Phase 5 — Drill/screw trajectory editor

Tasks:

- [ ] Add screw axis objects.
- [ ] Control entry point, direction, length, diameter.
- [ ] Visualize collision/intersection with bone.
- [ ] Export screw/drill JSON.
- [ ] Use trajectories to generate drill sleeves and plate holes.

## Phase 6 — Patient-specific plate generator

Tasks:

- [ ] Draw/select centerline on bone.
- [ ] Fit plate surface to anatomy.
- [ ] Width/thickness controls.
- [ ] Screw hole generator.
- [ ] Screw trajectory integration.
- [ ] Export STL initially; later STEP via FreeCAD/OpenCascade.

## Phase 7 — Robust CAD backend

Tasks:

- [ ] Integrate FreeCAD/OpenCascade for solids and STEP export.
- [ ] Add Blender fallback for mesh booleans.
- [ ] Add mesh repair pipeline.
- [ ] Validate exports with automated checks.

## Phase 8 — Case management / collaboration

Tasks:

- [ ] Cases, patients, clinicians, engineers.
- [ ] Upload/download storage.
- [ ] Audit log.
- [ ] Plan versions.
- [ ] Approval workflow.
- [ ] PDF surgical plan report.
- [ ] Portal integration.

## Non-goals for now

- Do not rebuild Horos.
- Do not prioritize website/marketing.
- Do not claim medical-device readiness.
