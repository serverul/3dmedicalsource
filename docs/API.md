# API Documentation — 3DMedicalPlanner

Base URL: `http://66.245.205.166:8121`

## Health
- `GET /api/health` — App status

## Import
- `POST /api/upload` — Upload STL/OBJ/PLY
- `POST /api/sample` — Generate synthetic bone sample
- `POST /api/upload-dicom` — Upload DICOM ZIP/.dcm → bone STL
- `POST /api/upload-dicom-folder` — Upload DICOM folder → bone STL

## Case Management
- `GET /api/recent-cases` — List recent runtime cases
- `GET /api/cases-list` — List all cases with metadata (supports `?status=draft`)
- `GET /api/cases/{case_id}/dicom-info` — DICOM series metadata
- `GET /api/cases/{case_id}/meta` — Case metadata + audit log
- `PUT /api/cases/{case_id}/meta` — Update case metadata
- `POST /api/cases/{case_id}/snapshot` — Create named snapshot
- `GET /api/cases/{case_id}/snapshots` — List snapshots
- `GET /api/cases/{case_id}/report` — Surgical plan report (JSON)

## Segmentation
- `POST /api/cases/{case_id}/regenerate-mesh` — Regenerate bone STL with full v2 parameters
  - `threshold_hu`, `threshold_max_hu`, `step_size`, `series_uid`, `keep_largest`, `smoothing_iterations`
  - `morphology_closing_radius`, `morphology_opening_radius`, `dilation_radius`, `erosion_radius`
  - `fill_holes`, `min_island_volume_mm3`, `roi_crop`, `pca_align`, `decimation_target_faces`
  - `preset` (long_bone | skull | spine | cortical | metal)

## MPR (Multi-Planar Reconstruction)
- `GET /api/cases/{case_id}/mpr-volinfo` — Volume dimensions, spacing, HU range
- `GET /api/cases/{case_id}/slice-range` — Min/max slice indices per axis
- `GET /api/cases/{case_id}/slice` — Extract slice as PNG
  - `axis` (axial|coronal|sagittal), `index`, `window_center`, `window_width`, `overlay_mask`

## Osteotomy
- `POST /api/cut` — Basic axis-aligned cut + fragment transform
- `POST /api/cut-from-points` — Cut from 3 points defining a plane
- `POST /api/cut-multi` — Multi-plane osteotomy (up to 3 planes)
- `GET /api/cases/{case_id}/fragment-transforms` — Get stored fragment transforms
- `POST /api/cases/{case_id}/fragment-transforms/{name}` — Update fragment transform
- `POST /api/export-fragments` — Export ZIP with all fragment STLs + plan JSON
- `POST /api/export-fragment-stl` — Export combined STL of selected fragments

## Measurement Tools
- `POST /api/measure-distance` — Distance between two points
- `POST /api/measure-angle` — Angle at vertex between two arms
- `POST /api/measure-bone-length` — Bone length along axis

## Guides
- `POST /api/guide` — Parametric cutting guide (rails + saw slot + pin sleeves)
- `POST /api/conformal-guide` — Conformal contact pad + slot rails
- `POST /api/contact-patch` — Contact patch from clicked point + radius
- `POST /api/plate-generator` — Patient-specific plate from centerline

## Screw Trajectories
- `GET /api/cases/{case_id}/screw-trajectories` — List all trajectories
- `POST /api/cases/{case_id}/screw-trajectories` — Add trajectory (entry + direction)
- `DELETE /api/cases/{case_id}/screw-trajectories/{traj_id}` — Remove trajectory
- `POST /api/cases/{case_id}/screw-trajectories/generate-sleeves` — Generate drill sleeves STL

## Mesh Operations
- `POST /api/mesh-boolean` — Boolean operation (difference/union/intersection)
- `POST /api/mirror-mesh` — Mirror mesh across axis

## Cephalometry
- `POST /api/cephalometry` — Compute angles from landmarks (S, N, A, B, Go, Me)
  - Returns: SNA, SNB, ANB, FMA, distances

## DICOM Anonymization
- `POST /api/cases/{case_id}/anonymize-dicom` — Anonymize DICOM files (26 tags)

## Undo/Redo
- `POST /api/cases/{case_id}/undo` — Undo last action
- `POST /api/cases/{case_id}/redo` — Redo last undone action
- `GET /api/cases/{case_id}/history` — Action history

## Files
- `GET /api/cases/{case_id}/input.stl` — Input/generated STL
- `GET /api/cases/{case_id}/outputs/{name}` — Output files
- `GET /api/cases/{case_id}/outputs/plan.json` — Cut plan JSON
- `GET /api/cases/{case_id}/outputs/osteotomy_export.zip` — Osteotomy export
