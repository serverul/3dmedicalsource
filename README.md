# 3DMedicalPlanner MVP

Research prototype for **patient-specific 3D surgical CAD**: CT/DICOM-derived bone meshes, virtual osteotomy, fragment repositioning, cutting guides, and first-pass conformal guide surfaces.

This is **not a certified medical device**. It is a research/MVP codebase intended for rapid iteration with agents and human validation.

## Current status

Working local web app:

- FastAPI backend (`app.py`)
- single-page Three.js UI (`static/index.html`)
- STL/OBJ/PLY upload
- DICOM ZIP upload
- full DICOM folder upload via browser `webkitdirectory`
- CT → HU volume → bone mesh STL via marching cubes
- **Segmentation Engine v2** with:
  - DICOM series selection
  - HU min/max thresholding
  - keep largest connected component
  - smoothing iterations
  - step size
  - **binary morphology**: closing, opening, dilation, erosion
  - **fill holes** (3D binary fill)
  - **remove small islands** by minimum physical volume (mm³)
  - **ROI auto-crop** around bone voxels
  - **PCA axis alignment** for CAD-friendly orientation
  - **mesh cleanup pipeline**: Laplacian smoothing, quadric decimation, normal/winding/hole repair
  - **segmentation presets**: long_bone, skull, spine, cortical, metal
  - **manifold/watertight validation report** in metadata
  - **segmentation parameters JSON** exported with every STL
  - regenerate mesh without re-upload
- 3D STL rendering in browser
- plane cut on X/Y/Z
- positive/negative fragment split
- capped/watertight cut when possible, fallback to open mesh
- rotate/translate selected fragment
- export positive/negative/combined STL and plan JSON
- parametric cutting guide STL with saw slot and pin sleeves
- first-pass conformal contact pad / conformal guide STL

## Quick start

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/uvicorn app:app --host 0.0.0.0 --port 8121
```

Open:

```txt
http://127.0.0.1:8121
```

For Vlad's Tailscale workflow, run on `0.0.0.0` and share the machine's Tailscale IP, not localhost.

## Verification

```bash
./venv/bin/python -m py_compile app.py
./venv/bin/python test_dicom.py
curl -fsS http://127.0.0.1:8121/api/health
```

Expected `test_dicom.py` result: non-zero vertices/faces from a synthetic DICOM CT series.

## Important data policy

Do **not** commit real CT/DICOM, patient data, generated case folders, or exported medical files.

Ignored paths:

```txt
data/uploads/
data/outputs/
data/tmp/
venv/
__pycache__/
```

Only the synthetic demo STL in `data/samples/sample_long_bone.stl` may be committed.

## Medical / regulatory warning

This prototype is for research and planning experiments only. It is not validated for clinical use, surgical execution, manufacturing, diagnosis, or treatment. Real patient-specific guides/plates require:

- validated segmentation workflow
- clinical review
- traceability and audit log
- versioned case files
- manufacturing validation
- sterilization/material validation
- regulatory pathway if used clinically

## Main workflow

```txt
CT/DICOM folder or ZIP
→ select DICOM series
→ choose HU segmentation range
→ generate bone STL
→ clean/smooth mesh
→ define osteotomy plane
→ split bone fragments
→ rotate/translate fragment
→ generate guide / conformal guide
→ export STL/JSON
```

## Current repository layout

```txt
.
├── app.py                    # FastAPI API + segmentation/mesh/CAD logic
├── static/index.html          # Three.js single-page UI
├── test_dicom.py              # synthetic DICOM regression test
├── check.py                   # local mesh cut diagnostic helper
├── requirements.txt
├── start.sh                   # local service starter used on Vlad's machine
├── data/
│   ├── samples/               # safe synthetic demo sample only
│   ├── uploads/               # ignored: uploaded DICOM/STL/case data
│   └── outputs/               # ignored: generated outputs
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── CURRENT_STATE.md
│   ├── RESEARCH_NOTES.md
│   └── ROADMAP.md
└── AGENTS.md
```

## Core dependencies

- `fastapi`, `uvicorn`, `python-multipart`
- `pydicom`
- `numpy`, `scipy`, `scikit-image`
- `trimesh`, `shapely`, `mapbox-earcut`
- Three.js loaded from CDN in the browser

## What to work on next

See [`docs/ROADMAP.md`](docs/ROADMAP.md). Highest priority now:

1. Segmentation Engine v2: morphology, fill holes, island removal, ROI crop, axis alignment.
2. Robust CT viewer/segmentation preview instead of blind 3D mesh generation.
3. Interactive contact patch selection for conformal guides.
4. Drill/screw trajectory editor.
5. Plate centerline + patient-specific plate generator.
