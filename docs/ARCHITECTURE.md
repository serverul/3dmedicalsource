# Architecture

## Overview

3DMedicalPlanner is currently a single-process MVP:

```txt
Browser UI (Three.js)
        │
        ▼
FastAPI backend (`app.py`)
        │
        ├── DICOM import / segmentation
        ├── mesh slicing / transforms
        ├── guide generation
        └── runtime data in `data/`
```

No database, auth, job queue, or persistent case model exists yet. Case IDs are directory names generated from UUID fragments.

## Runtime paths

```python
BASE = project root
DATA = BASE / "data"
UPLOADS = DATA / "uploads"
OUTPUTS = DATA / "outputs"
SAMPLES = DATA / "samples"
STATIC = BASE / "static"
```

- `data/uploads/{case_id}` stores uploaded input and DICOM folders.
- `data/outputs/{case_id}` stores generated STL/JSON outputs.
- These are ignored because they can contain PHI.

## Frontend

`static/index.html` includes:

- Three.js renderer
- `STLLoader`
- `OrbitControls`
- import controls for STL/OBJ/PLY/DICOM
- segmentation controls
- cut/transform controls
- guide/conformal-guide controls

Three.js is loaded from CDN via importmap. This is fine for MVP, but production should vendor/lock frontend dependencies.

## Backend modules inside `app.py`

### Mesh loading/info

- `load_mesh(path)`
- `mesh_info(mesh)`

### Geometry/CAD

- `axis_normal(axis)`
- `safe_slice(mesh, normal, origin, cap=True)`
- `split_mesh(mesh, axis, offset_mm)`
- `rotation_matrix(axis, degrees)`
- `transform_mesh(mesh, axis, degrees, translate)`
- `generate_cutting_guide(...)`
- `generate_conformal_pad(...)`

### DICOM/segmentation

- `dicom_spacing_and_position(ds)`
- `load_dicom_series(dicom_dir, selected_series_uid=None)`
- `dicom_to_bone_mesh(...)`

Current DICOM flow:

```txt
read all files recursively
→ pydicom dcmread(force=True)
→ keep 2D PixelData images
→ group by SeriesInstanceUID
→ select requested/largest series
→ sort by ImagePositionPatient z / InstanceNumber fallback
→ apply slope/intercept to HU
→ HU threshold / optional HU range mask
→ optional keep-largest connected component
→ marching_cubes with real spacing
→ convert z,y,x to x,y,z
→ optional smoothing
→ export STL
```

## Data model

There is no DB yet. A case is represented by directories:

```txt
data/uploads/{case_id}/
  input.stl
  dicom/...
  dicom_meta.json

data/outputs/{case_id}/
  positive.stl
  negative.stl
  combined_repositioned.stl
  plan.json
  cutting_guide.stl
  guide.json
  conformal_contact_pad.stl
  conformal_cutting_guide.stl
  conformal_guide.json
```

## Production direction

For a real product split this into:

```txt
frontend/      React/Next.js or proper Vite app
backend/       FastAPI API
worker/        DICOM/mesh/CAD jobs
cad-engine/    FreeCAD/OpenCascade/Blender runners
storage/       MinIO/S3
postgres/      cases/users/audit/versioning
```

## Design constraints

- STL is acceptable for prototype guides/models.
- Plates/metal implants likely need STEP/solid CAD: FreeCAD/OpenCascade.
- Mesh booleans on segmented anatomy are fragile; use repair/fallback and expose validation.
- Browser should become UI/preview; heavy CAD should run server/local worker.
