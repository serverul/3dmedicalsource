# Current State

Last updated: 2026-05-23 (Segmentation Engine v2).

## What has been built

### Application shell

- FastAPI backend in `app.py`.
- Static Three.js UI in `static/index.html`.
- Local service starter `start.sh`.
- Runtime on Vlad's machine used port `8121`.

### Import

- STL/OBJ/PLY upload.
- DICOM ZIP upload.
- Single `.dcm` upload, although a single slice is generally not enough for 3D reconstruction.
- DICOM folder upload using browser `webkitdirectory` and multipart list of files.

### DICOM → mesh

- Reads DICOM slices with `pydicom`.
- Groups by `SeriesInstanceUID`.
- Chooses largest series by default.
- Preserves spacing from `PixelSpacing`, `ImagePositionPatient`, and `SliceThickness` fallback.
- Converts pixel values to HU using `RescaleSlope` and `RescaleIntercept`.
- Marching cubes via `skimage.measure.marching_cubes`.
- Converts skimage `z,y,x` vertices into CAD-like `x,y,z`.

### Segmentation controls

- Series selector.
- HU minimum.
- HU maximum / metal cutoff.
- Step size.
- Keep largest component.
- Smoothing iterations.
- Regenerate mesh for an existing DICOM case without re-upload.

### CAD prototype

- Plane selection: X/Y/Z and offset in mm.
- Split mesh into positive/negative fragments.
- Attempts capped/watertight cuts using `trimesh`; falls back to open cut.
- Rotate selected fragment around X/Y/Z.
- Translate selected fragment in X/Y/Z.
- Export positive, negative, and combined repositioned STL.
- Save plan JSON.

### Guides

- Parametric cutting guide:
  - two rails
  - saw slot
  - pin sleeves
  - STL + JSON export
- First conformal guide MVP:
  - samples bone surface locally using KD-tree
  - creates contact pad following anatomy approximation
  - adds thickness
  - optionally concatenates with slot rails
  - STL + JSON export

## Known limitations

- DICOM segmentation is still threshold-based; morphology helps but is not AI/ML.
- No 2D CT slice viewer yet; segmentation is mostly blind except for resulting 3D STL.
- ROI crop adjusts mesh position — CAD operations after crop use the new origin.
- PCA alignment changes mesh orientation; disable if original orientation matters.
- Decimation is quadric — may lose fine detail on small anatomical features.
- Mesh repair is best-effort; severely broken meshes may still fail watertight check.
- `keep_largest` currently works best when HU max creates a binary mask path.
- No manual segmentation editing.
- Conformal guide is approximate, not a true boolean negative mold.
- Cutting guides/pads are not clinically validated.
- No screw trajectory editor.
- No patient-specific plate generator yet.
- No auth/case management/audit trail.

## Important CT test observed during development

A real uploaded CT case had one detected series:

- series: `Toraco/Lombare 1.00 Br40 S3`
- slices: `16`
- spacing: approximately `[0.7, 0.201, 0.201]` mm
- HU max very high (`~29621`), likely metal/artifact

Using HU range `250–3000`, keep-largest, and smoothing produced a smaller cleaner mesh than raw threshold.

Do not commit this case or any DICOM files; it may contain patient data.
