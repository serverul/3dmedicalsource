# Roadmap — 3DMedicalPlanner

## Product direction

3DMedicalPlanner is not a generic DICOM viewer. The target is:

```txt
3D CT-based patient-specific surgical CAD for bones
```

Core value:
- extract accurate bone model from CT
- virtual osteotomy
- fragment repositioning
- patient-specific cutting/drill guides
- patient-specific plates/implants
- exportable manufacturing files

## Phase 1 — Segmentation Engine v2 ✅ COMPLETED

- [x] Binary morphology: closing, opening, dilation, erosion, fill holes
- [x] Remove small islands by physical volume (mm³)
- [x] ROI auto-crop around bone voxels
- [x] PCA axis alignment
- [x] Mesh cleanup: smoothing, decimation, repair, manifold report
- [x] Segmentation presets: long_bone, skull, spine, cortical, metal
- [x] Export segmentation parameters JSON with every STL

## Phase 2 — MPR Slice Viewer ✅ COMPLETED

- [x] Axial/coronal/sagittal slice extraction as PNG
- [x] HU window/level controls
- [x] Segmentation mask overlay on slices
- [x] Volume info and slice range endpoints
- [x] Volume caching (.npy) for fast access

## Phase 3 — Advanced Osteotomy Planning ✅ COMPLETED

- [x] Interactive 3-point plane definition
- [x] Multi-plane osteotomy (up to 3 planes)
- [x] Named fragments with color coding
- [x] Fragment transform editor (translate/rotate per fragment)
- [x] Measurement tools: distance, angle, bone length
- [x] Export per-fragment STLs + combined plan JSON
- [x] Export ZIP with all fragments

## Phase 4 — Conformal Surgical Guides ✅ COMPLETED

- [x] Interactive contact patch selection (radius-based)
- [x] Conformal pad generation from patch
- [x] Mesh boolean operations (difference/union/intersection)
- [x] Guide body solidification
- [x] Saw slot from osteotomy plane
- [x] Drill/pin sleeves

## Phase 5 — Drill/Screw Trajectory Editor ✅ COMPLETED

- [x] Add screw trajectories (entry point + direction)
- [x] Control length, diameter, name
- [x] Collision detection with bone
- [x] Generate drill guide sleeves STL
- [x] CRUD operations on trajectories

## Phase 6 — Patient-Specific Plate Generator ✅ COMPLETED

- [x] Centerline-based plate generation
- [x] Width/thickness controls
- [x] Automatic screw hole placement
- [x] Boolean subtraction of screw holes
- [x] Export STL

## Phase 7 — Undo/Redo System ✅ COMPLETED

- [x] Per-case undo/redo stack (max 50 actions)
- [x] Snapshot-based state restoration
- [x] Action history with timestamps
- [x] Keyboard shortcuts (Ctrl+Z / Ctrl+Shift+Z)
- [x] Persistence to disk

## Phase 8 — Case Management & Utilities ✅ COMPLETED

- [x] Case metadata CRUD (patient_id, description, status)
- [x] Audit log (auto-tracked on every operation)
- [x] Named snapshots of case state
- [x] Surgical plan report (JSON)
- [x] DICOM anonymization (26 tags)
- [x] Mesh mirror/reflection
- [x] Cephalometry (SNA, SNB, ANB, FMA, distances)

## Future Enhancements

- [ ] AI-assisted segmentation (cloud-based model)
- [ ] PACS integration
- [ ] Multi-user collaboration (SignalR real-time)
- [ ] PDF report generation
- [ ] STEP export via FreeCAD/OpenCascade
- [ ] Blender fallback for mesh booleans
- [ ] User authentication
- [ ] Cloud storage sync
