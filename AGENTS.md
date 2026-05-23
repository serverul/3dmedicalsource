# AGENTS.md — 3DMedicalPlanner

You are working on 3DMedicalPlanner, a research prototype for 3D CT-based surgical CAD.

## Mission

Build a practical workflow for patient-specific bone planning:

```txt
DICOM CT → bone segmentation → clean STL → virtual osteotomy → fragment repositioning → cutting guide / drill guide / plate → exportable manufacturing files
```

Do not spend time on marketing websites or generic 2D viewers. The value is surgical CAD.

## Current app

- Backend: `app.py` FastAPI
- UI: `static/index.html` using Three.js
- Runtime data: `data/uploads` and `data/outputs` — ignored, may contain PHI. Never commit.
- Synthetic test: `test_dicom.py`

## Ground rules

1. **Never commit real CT/DICOM/patient data.** Check `git status --ignored` if unsure.
2. Treat all geometry as millimeters. Preserve DICOM spacing correctly.
3. Prototype status only — never claim clinical/manufacturing validation.
4. Prefer small, verifiable steps. Add tests or CLI checks for every algorithmic change.
5. Keep README/docs updated after meaningful changes so other agents can continue.
6. When adding CAD features, export both geometry and JSON parameters for reproducibility.
7. Report watertight/manifold status when generating STL outputs.

## How to run

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8121
```

Verify:

```bash
./venv/bin/python -m py_compile app.py
./venv/bin/python test_dicom.py
curl -fsS http://127.0.0.1:8121/api/health
```

## Immediate next tasks

See `docs/ROADMAP.md`. Current technical focus:

- Segmentation Engine v2: fill holes, morphology, ROI crop, axis alignment.
- Better DICOM/segmentation preview before mesh generation.
- Interactive contact patch selection for conformal guide.
- Drill/screw trajectories.
- Patient-specific plate centerline and holes.

## Useful implementation notes

- `mapbox-earcut` is needed for capped `trimesh` slicing.
- Mesh booleans on CT-derived anatomy are fragile. Always implement fallbacks.
- `trimesh.intersections.slice_mesh_plane(..., cap=True)` works on clean sample meshes but can fail on messy medical meshes.
- Browser folder upload uses `webkitdirectory`; keep the ZIP/.dcm fallback.
- DICOM folders may contain multiple series. Always expose series choice.
