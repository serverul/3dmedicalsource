import json
import math
import os
import shutil
import time
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import binary_closing, binary_opening, binary_dilation, binary_erosion, label as ndi_label
from skimage import measure
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from undo_manager import UndoManager, UndoAction, get_manager  # noqa: E402

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
UPLOADS = DATA / "uploads"
OUTPUTS = DATA / "outputs"
SAMPLES = DATA / "samples"
STATIC = BASE / "static"
for p in (UPLOADS, OUTPUTS, SAMPLES, STATIC):
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="3DMedicalPlanner MVP", version="0.1.0")


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError("Invalid or empty mesh")
    mesh.remove_unreferenced_vertices()
    return mesh


def mesh_info(mesh: trimesh.Trimesh) -> dict:
    bounds = mesh.bounds.tolist()
    ext = mesh.extents.tolist()
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "bounds": bounds,
        "extents_mm": ext,
        "volume_mm3": float(mesh.volume) if mesh.is_watertight else None,
        "area_mm2": float(mesh.area),
    }


def axis_normal(axis: str) -> np.ndarray:
    axis = axis.lower()
    if axis == "x":
        return np.array([1.0, 0.0, 0.0])
    if axis == "y":
        return np.array([0.0, 1.0, 0.0])
    return np.array([0.0, 0.0, 1.0])


def safe_slice(mesh: trimesh.Trimesh, normal: np.ndarray, origin: np.ndarray, cap: bool = True) -> trimesh.Trimesh:
    # Prefer capped/watertight fragments when triangulation is available; fall back to open cuts.
    last_error = None
    for do_cap in ([True, False] if cap else [False]):
        try:
            sliced = trimesh.intersections.slice_mesh_plane(
                mesh,
                plane_normal=normal,
                plane_origin=origin,
                cap=do_cap,
            )
            if sliced is None or len(sliced.faces) == 0:
                continue
            sliced.remove_unreferenced_vertices()
            return sliced
        except Exception as e:
            last_error = e
            continue
    return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = normal / np.linalg.norm(normal)
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    return n, u, v


def basis_transform(origin: np.ndarray, n: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, 0] = n
    mat[:3, 1] = u
    mat[:3, 2] = v
    mat[:3, 3] = origin
    return mat


def create_tube(radius_outer: float, radius_inner: float, length: float, sections: int = 48) -> trimesh.Trimesh:
    # Local tube along X axis, centered at origin. Watertight ring/sleeve mesh.
    xs = [-length / 2.0, length / 2.0]
    verts = []
    for x in xs:
        for r in (radius_outer, radius_inner):
            for i in range(sections):
                a = 2 * math.pi * i / sections
                verts.append([x, r * math.cos(a), r * math.sin(a)])
    faces = []
    def idx(end, ring, i):
        return end * sections * 2 + ring * sections + (i % sections)
    for i in range(sections):
        j = i + 1
        # outer wall
        faces.append([idx(0, 0, i), idx(1, 0, i), idx(1, 0, j)])
        faces.append([idx(0, 0, i), idx(1, 0, j), idx(0, 0, j)])
        # inner wall reversed
        faces.append([idx(0, 1, i), idx(1, 1, j), idx(1, 1, i)])
        faces.append([idx(0, 1, i), idx(0, 1, j), idx(1, 1, j)])
        # end caps as annulus faces
        faces.append([idx(0, 0, i), idx(0, 0, j), idx(0, 1, j)])
        faces.append([idx(0, 0, i), idx(0, 1, j), idx(0, 1, i)])
        faces.append([idx(1, 0, i), idx(1, 1, j), idx(1, 0, j)])
        faces.append([idx(1, 0, i), idx(1, 1, i), idx(1, 1, j)])
    return trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=True)


def generate_conformal_pad(
    mesh: trimesh.Trimesh,
    axis: str,
    offset_mm: float,
    contact_side: str,
    length_mm: float,
    height_mm: float,
    thickness_mm: float,
    clearance_mm: float,
    resolution_u: int = 28,
    resolution_v: int = 14,
) -> tuple[trimesh.Trimesh, dict]:
    """Create a watertight pad whose inner surface approximates the local bone surface.

    This is a pragmatic MVP algorithm: project mesh vertices into the cut-plane basis,
    sample a rectangular footprint, and use nearest local vertices to estimate the
    visible surface on the selected side. It is not a certified negative mold yet,
    but it produces a real curved contact surface from patient mesh geometry.
    """
    if resolution_u < 3 or resolution_v < 3:
        raise ValueError("resolution too low")
    normal = axis_normal(axis)
    center = mesh.bounds.mean(axis=0)
    origin = center + normal * offset_mm
    n, u, v = plane_basis(normal)
    out_sign = 1.0 if contact_side.lower() != "negative" else -1.0

    rel = mesh.vertices - origin
    local_x = rel @ n
    local_y = rel @ u
    local_z = rel @ v
    yz = np.column_stack([local_y, local_z])
    tree = cKDTree(yz)

    ys = np.linspace(-length_mm / 2.0, length_mm / 2.0, resolution_u)
    zs = np.linspace(-height_mm / 2.0, height_mm / 2.0, resolution_v)
    search_k = min(64, len(mesh.vertices))

    inner = []
    outer = []
    sampled = []
    for z in zs:
        for y in ys:
            _dist, idxs = tree.query([y, z], k=search_k)
            idxs = np.atleast_1d(idxs)
            xs = local_x[idxs]
            if out_sign > 0:
                surface_x = float(np.max(xs))
            else:
                surface_x = float(np.min(xs))
            # Light smoothing against wild outliers: blend with median extreme neighborhood.
            surface_x = float(0.75 * surface_x + 0.25 * np.median(xs))
            inner_x = surface_x + out_sign * clearance_mm
            outer_x = inner_x + out_sign * thickness_mm
            inner_pt = origin + n * inner_x + u * y + v * z
            outer_pt = origin + n * outer_x + u * y + v * z
            inner.append(inner_pt)
            outer.append(outer_pt)
            sampled.append(surface_x)

    verts = np.vstack([np.array(inner), np.array(outer)])
    faces = []
    nu = len(ys)
    nv = len(zs)
    inner_off = 0
    outer_off = nu * nv

    def gid(layer_off: int, row: int, col: int) -> int:
        return layer_off + row * nu + col

    for r in range(nv - 1):
        for c in range(nu - 1):
            a = gid(inner_off, r, c); b = gid(inner_off, r, c + 1); c1 = gid(inner_off, r + 1, c + 1); d = gid(inner_off, r + 1, c)
            ao = gid(outer_off, r, c); bo = gid(outer_off, r, c + 1); co = gid(outer_off, r + 1, c + 1); do = gid(outer_off, r + 1, c)
            # Inner surface faces inward toward bone; outer faces outward.
            if out_sign > 0:
                faces.extend([[a, c1, b], [a, d, c1], [ao, bo, co], [ao, co, do]])
            else:
                faces.extend([[a, b, c1], [a, c1, d], [ao, co, bo], [ao, do, co]])

    # Boundary walls
    for r in range(nv - 1):
        for col in (0, nu - 1):
            a = gid(inner_off, r, col); b = gid(inner_off, r + 1, col); bo = gid(outer_off, r + 1, col); ao = gid(outer_off, r, col)
            faces.extend([[a, b, bo], [a, bo, ao]])
    for c in range(nu - 1):
        for row in (0, nv - 1):
            a = gid(inner_off, row, c); b = gid(inner_off, row, c + 1); bo = gid(outer_off, row, c + 1); ao = gid(outer_off, row, c)
            faces.extend([[a, bo, b], [a, ao, bo]])

    pad = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    pad.remove_unreferenced_vertices()
    meta = {
        "axis": axis,
        "offset_mm": offset_mm,
        "contact_side": "positive" if out_sign > 0 else "negative",
        "length_mm": length_mm,
        "height_mm": height_mm,
        "thickness_mm": thickness_mm,
        "clearance_mm": clearance_mm,
        "resolution": [resolution_u, resolution_v],
        "sampled_surface_x_mm": {"min": float(np.min(sampled)), "max": float(np.max(sampled)), "mean": float(np.mean(sampled))},
        "note": "Conformal MVP: inner surface is estimated from nearest mesh vertices in the local cut-plane footprint.",
    }
    return pad, meta


def generate_cutting_guide(
    mesh: trimesh.Trimesh,
    axis: str,
    offset_mm: float,
    length_mm: float,
    height_mm: float,
    rail_thickness_mm: float,
    slot_width_mm: float,
    pin_radius_mm: float,
    pin_spacing_mm: float,
    pin_sleeve_outer_mm: float,
) -> tuple[trimesh.Trimesh, dict]:
    normal = axis_normal(axis)
    center = mesh.bounds.mean(axis=0)
    origin = center + normal * offset_mm
    n, u, v = plane_basis(normal)

    parts = []
    # Two guide rails separated by the saw slot centered on the cutting plane.
    for sign in (-1, 1):
        rail_center = origin + n * sign * (slot_width_mm / 2.0 + rail_thickness_mm / 2.0)
        box = trimesh.creation.box(extents=[rail_thickness_mm, length_mm, height_mm])
        box.apply_transform(basis_transform(rail_center, n, u, v))
        parts.append(box)

    sleeve_len = rail_thickness_mm * 2 + slot_width_mm
    sleeve_outer = max(pin_sleeve_outer_mm, pin_radius_mm + 1.2)
    for y in (-pin_spacing_mm / 2.0, pin_spacing_mm / 2.0):
        sleeve = create_tube(sleeve_outer, pin_radius_mm, sleeve_len, sections=48)
        sleeve_origin = origin + u * y
        sleeve.apply_transform(basis_transform(sleeve_origin, n, u, v))
        parts.append(sleeve)

    guide = trimesh.util.concatenate(parts)
    guide.remove_unreferenced_vertices()
    meta = {
        "plane_axis": axis,
        "plane_origin_mm": origin.tolist(),
        "plane_normal": n.tolist(),
        "length_mm": length_mm,
        "height_mm": height_mm,
        "rail_thickness_mm": rail_thickness_mm,
        "slot_width_mm": slot_width_mm,
        "pin_radius_mm": pin_radius_mm,
        "pin_spacing_mm": pin_spacing_mm,
        "pin_sleeve_outer_mm": sleeve_outer,
        "note": "Parametric first-pass cutting guide: two rails create the saw slot; pin sleeves are hollow tubes. Contact surface is not yet conformal to bone.",
    }
    return guide, meta


def rotate_mesh(mesh: trimesh.Trimesh, axis: str, degrees: float, pivot: np.ndarray) -> trimesh.Trimesh:
    out = mesh.copy()
    if abs(degrees) < 1e-9 or len(out.vertices) == 0:
        return out
    normal = axis_normal(axis)
    mat = trimesh.transformations.rotation_matrix(math.radians(degrees), normal, point=pivot)
    out.apply_transform(mat)
    return out


def translate_mesh(mesh: trimesh.Trimesh, dx: float, dy: float, dz: float) -> trimesh.Trimesh:
    out = mesh.copy()
    if len(out.vertices) and (dx or dy or dz):
        out.apply_translation([dx, dy, dz])
    return out


def dicom_spacing_and_position(ds):
    ps = [float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])]
    ipp = getattr(ds, "ImagePositionPatient", None)
    z = float(ipp[2]) if ipp is not None and len(ipp) >= 3 else float(getattr(ds, "InstanceNumber", 0))
    return ps, z


def load_dicom_series(dicom_dir: Path, selected_series_uid: Optional[str] = None) -> tuple[np.ndarray, tuple[float, float, float], dict]:
    by_series = {}
    series_desc = {}
    for p in dicom_dir.rglob("*"):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(p), force=True)
            if not hasattr(ds, "PixelData"):
                continue
            arr = ds.pixel_array  # may fail for compressed unsupported transfer syntaxes
            if arr.ndim != 2:
                continue
            _ps, z = dicom_spacing_and_position(ds)
            uid = str(getattr(ds, "SeriesInstanceUID", "NO_SERIES_UID"))
            by_series.setdefault(uid, []).append((z, ds, arr.astype(np.float32)))
            series_desc[uid] = str(getattr(ds, "SeriesDescription", ""))
        except Exception:
            continue
    if not by_series:
        raise ValueError("No readable DICOM slices with PixelData found. For compressed DICOM, install decoder or export uncompressed CT.")
    # Pick requested series, otherwise largest CT series in uploaded folder.
    if selected_series_uid and selected_series_uid in by_series:
        series_uid, datasets = selected_series_uid, by_series[selected_series_uid]
    else:
        series_uid, datasets = max(by_series.items(), key=lambda kv: len(kv[1]))
    if len(datasets) < 2:
        raise ValueError("Need at least 2 readable DICOM slices in one series for 3D reconstruction.")
    datasets.sort(key=lambda x: x[0])
    zs = [x[0] for x in datasets]
    ds0 = datasets[0][1]
    ps, _ = dicom_spacing_and_position(ds0)
    dzs = np.diff(zs)
    slice_spacing = float(np.median(np.abs(dzs))) if len(dzs) and np.median(np.abs(dzs)) > 0 else float(getattr(ds0, "SliceThickness", 1.0))

    volume = []
    for _z, ds, arr in datasets:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        volume.append(arr * slope + intercept)
    vol = np.stack(volume, axis=0).astype(np.float32)  # z, rows, cols
    meta = {
        "slices": len(datasets),
        "series_uid": series_uid,
        "series_description": series_desc.get(series_uid, ""),
        "available_series": [{"uid": uid, "description": series_desc.get(uid, ""), "slices": len(rows)} for uid, rows in by_series.items()],
        "rows": int(vol.shape[1]),
        "cols": int(vol.shape[2]),
        "spacing_mm": [slice_spacing, float(ps[0]), float(ps[1])],
        "patient_id": str(getattr(ds0, "PatientID", "")),
        "patient_name": str(getattr(ds0, "PatientName", "")),
        "study_description": str(getattr(ds0, "StudyDescription", "")),
        "modality": str(getattr(ds0, "Modality", "")),
        "hu_min": float(np.min(vol)),
        "hu_max": float(np.max(vol)),
    }
    return vol, (slice_spacing, float(ps[0]), float(ps[1])), meta


SEGMENTATION_PRESETS = {
    "whole_limb": {
        "threshold_hu": 150,
        "threshold_max_hu": 2500,
        "morphology_closing_radius": 3,
        "morphology_opening_radius": 1,
        "fill_holes": True,
        "min_island_volume_mm3": 50.0,
        "smoothing_iterations": 3,
        "decimation_target_faces": None,
    },
    "long_bone": {
        "threshold_hu": 200,
        "threshold_max_hu": 3000,
        "morphology_closing_radius": 3,
        "morphology_opening_radius": 1,
        "fill_holes": True,
        "min_island_volume_mm3": 500.0,
        "smoothing_iterations": 3,
        "decimation_target_faces": None,
    },
    "skull": {
        "threshold_hu": 180,
        "threshold_max_hu": 3000,
        "morphology_closing_radius": 4,
        "morphology_opening_radius": 2,
        "fill_holes": True,
        "min_island_volume_mm3": 200.0,
        "smoothing_iterations": 4,
        "decimation_target_faces": None,
    },
    "spine": {
        "threshold_hu": 200,
        "threshold_max_hu": 3000,
        "morphology_closing_radius": 2,
        "morphology_opening_radius": 1,
        "fill_holes": False,
        "min_island_volume_mm3": 200.0,
        "smoothing_iterations": 2,
        "decimation_target_faces": None,
    },
    "pelvis": {
        "threshold_hu": 200,
        "threshold_max_hu": 3000,
        "morphology_closing_radius": 3,
        "morphology_opening_radius": 1,
        "fill_holes": True,
        "min_island_volume_mm3": 500.0,
        "smoothing_iterations": 3,
        "decimation_target_faces": None,
    },
    "rib": {
        "threshold_hu": 180,
        "threshold_max_hu": 3000,
        "morphology_closing_radius": 2,
        "morphology_opening_radius": 2,
        "fill_holes": False,
        "min_island_volume_mm3": 100.0,
        "smoothing_iterations": 2,
        "decimation_target_faces": None,
    },
    "cortical": {
        "threshold_hu": 400,
        "threshold_max_hu": 3000,
        "morphology_closing_radius": 1,
        "morphology_opening_radius": 0,
        "fill_holes": True,
        "min_island_volume_mm3": 200.0,
        "smoothing_iterations": 1,
        "decimation_target_faces": None,
    },
    "metal": {
        "threshold_hu": 200,
        "threshold_max_hu": 8000,
        "morphology_closing_radius": 3,
        "morphology_opening_radius": 2,
        "fill_holes": True,
        "min_island_volume_mm3": 300.0,
        "smoothing_iterations": 2,
        "decimation_target_faces": None,
    },
}


def _auto_threshold(vol: np.ndarray, sensitivity: str = "medium") -> tuple[float, float, list[str]]:
    """Automatically determine optimal HU threshold for bone segmentation.

    Uses multi-peak histogram analysis inspired by Bonnet (arXiv 2601.22576):
    - Detects air peak (~-1000 HU), soft tissue peak (~0-100 HU), and bone peak (~300-1000 HU)
    - Finds the valley between soft tissue and bone peaks as the threshold
    - Falls back to Otsu's method on the relevant range if valley detection fails
    - Uses the 99.9th percentile of bone voxels as threshold_max, capped at 3500 HU

    sensitivity: "low" = higher threshold (less noise, may miss small/osteopenic bones)
                  "medium" = balanced (default)
                  "high" = lower threshold (captures more, including small bones)

    Returns (threshold_min, threshold_max, warnings).
    """
    warnings: list[str] = []
    flat = vol.flatten().astype(np.float32)

    # Focus on relevant HU range: -1100 to 4000
    relevant = flat[(flat > -1100) & (flat < 4000)]
    if len(relevant) == 0:
        warnings.append("No voxels in relevant HU range; using defaults")
        return 150.0, 2500.0, warnings

    total_voxels = float(len(relevant))

    # Build histogram with fine bins over the full range
    hist, bin_edges = np.histogram(relevant, bins=512, range=(-1100, 4000))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Smooth histogram for robust peak/valley detection
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(hist.astype(float), size=7)

    # --- Detect peaks ---
    # Soft tissue peak: maximum in range -50..150 HU
    st_mask = (bin_centers >= -50) & (bin_centers <= 150)
    # Bone peak: maximum in range 150..1500 HU (wider range for small animal bones)
    bone_peak_mask = (bin_centers >= 150) & (bin_centers <= 1500)

    st_peak_hu = 50.0
    if st_mask.any():
        st_idx = np.argmax(smoothed[st_mask])
        st_bins = bin_centers[st_mask]
        st_peak_hu = float(st_bins[st_idx])

    bone_peak_hu = 400.0
    if bone_peak_mask.any():
        bp_idx = np.argmax(smoothed[bone_peak_mask])
        bp_bins = bin_centers[bone_peak_mask]
        bone_peak_hu = float(bp_bins[bp_idx])

    # --- Find valley between soft tissue and bone peaks ---
    # Adjust valley range based on sensitivity
    sensitivity_offset = {"low": 80.0, "medium": 40.0, "high": 20.0}[sensitivity]
    valley_lo = max(st_peak_hu + sensitivity_offset, 50.0)
    valley_hi = min(bone_peak_hu - sensitivity_offset, 500.0)
    if valley_lo >= valley_hi:
        valley_lo = 50.0
        valley_hi = min(bone_peak_hu - 20.0, 500.0)
        if valley_lo >= valley_hi:
            valley_lo = 50.0
            valley_hi = 300.0

    valley_search = (bin_center

... [OUTPUT TRUNCATED - 90423 chars omitted out of 140423 total] ...

n operation with fallback.

    Operations: difference, union, intersection.
    Uses trimesh boolean (which uses manifold or blender as backend).
    Falls back to trimesh convention if boolean fails.
    """
    try:
        if operation == "difference":
            result = mesh_a.difference(mesh_b)
        elif operation == "union":
            result = mesh_a.union(mesh_b)
        elif operation == "intersection":
            result = mesh_a.intersection(mesh_b)
        else:
            raise ValueError(f"Unknown boolean operation: {operation}")
        if result is None or len(result.faces) == 0:
            raise ValueError("boolean result is empty")
        result.remove_unreferenced_vertices()
        return result
    except Exception:
        # Fallback: return mesh_a unchanged
        return mesh_a.copy()


@app.post("/api/mesh-boolean")
def mesh_boolean(
    case_id: str,
    operation: str = Form("difference"),  # difference | union | intersection
    fragment_a: str = Form("positive"),
    fragment_b: str = Form("cutting_guide"),
):
    """Perform boolean operation between two meshes from the case."""
    out_dir = OUTPUTS / case_id
    in_path = UPLOADS / case_id / "input.stl"

    # Load mesh A
    path_a = out_dir / f"fragment_{fragment_a}.stl"
    if not path_a.exists():
        path_a = out_dir / f"{fragment_a}.stl"
    if not path_a.exists() and fragment_a == "input":
        path_a = in_path
    if not path_a.exists():
        raise HTTPException(404, f"mesh A '{fragment_a}' not found")

    # Load mesh B
    path_b = out_dir / f"{fragment_b}.stl"
    if not path_b.exists():
        raise HTTPException(404, f"mesh B '{fragment_b}' not found")

    mesh_a = load_mesh(path_a)
    mesh_b = load_mesh(path_b)

    try:
        result = _mesh_boolean(mesh_a, mesh_b, operation)
    except Exception as e:
        raise HTTPException(400, f"boolean operation failed: {e}")

    ts = uuid.uuid4().hex[:8]
    out_name = f"boolean_{operation}_{fragment_a}_vs_{fragment_b}_{ts}.stl"
    out_path = out_dir / out_name
    result.export(out_path)

    return {
        "case_id": case_id,
        "operation": operation,
        "mesh_a": fragment_a,
        "mesh_b": fragment_b,
        "url": f"/api/cases/{case_id}/outputs/{out_name}",
        "info": mesh_info(result),
    }


# ══════════════════════════════════════════════════════════════════════
# Fase 4 — Interactive contact patch selection
# ══════════════════════════════════════════════════════════════════════

def _select_contact_patch(
    mesh: trimesh.Trimesh,
    center_point: np.ndarray,
    radius_mm: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Select mesh vertices within radius of a center point.

    Returns (face_indices, vertices) of the contact patch.
    """
    tree = cKDTree(mesh.vertices)
    vert_indices = tree.query_ball_point(center_point, r=radius_mm)
    if not vert_indices:
        # Fallback: nearest vertex
        dist, nearest = tree.query(center_point, k=1)
        vert_indices = [int(nearest)]

    # Find faces that use these vertices
    vert_set = set(vert_indices)
    face_mask = np.array([
        any(int(v) in vert_set for v in face)
        for face in mesh.faces
    ])
    face_indices = np.where(face_mask)[0]
    return face_indices, np.array(vert_indices)


@app.post("/api/contact-patch")
def contact_patch(
    case_id: str,
    center: str = Form(...),  # JSON [x,y,z]
    radius_mm: float = Form(30.0),
    thickness_mm: float = Form(4.0),
    clearance_mm: float = Form(0.6),
):
    """Generate a conformal contact pad from a selected patch on the bone surface."""
    in_path = UPLOADS / case_id / "input.stl"
    if not in_path.exists():
        raise HTTPException(404, "case not found")

    try:
        center_pt = np.array(json.loads(center), dtype=float)
    except Exception:
        raise HTTPException(400, "invalid center point")

    mesh = load_mesh(in_path)
    face_indices, vert_indices = _select_contact_patch(mesh, center_pt, radius_mm)

    if len(face_indices) == 0:
        raise HTTPException(400, "no mesh faces found in the selected patch")

    # Extract sub-mesh for the patch
    patch_faces = mesh.faces[face_indices]
    used_verts = np.unique(patch_faces)
    remap = {int(v): i for i, v in enumerate(used_verts)}
    new_faces = np.vectorize(remap.get)(patch_faces).astype(np.int64)
    patch_mesh = trimesh.Trimesh(
        vertices=mesh.vertices[used_verts],
        faces=new_faces,
        process=True,
    )
    patch_mesh.remove_unreferenced_vertices()

    # Generate conformal pad using the existing function
    # Use the patch center to determine the best axis
    normal = patch_mesh.face_normals.mean(axis=0)
    norm_len = np.linalg.norm(normal)
    if norm_len > 1e-12:
        normal = normal / norm_len
    else:
        normal = np.array([0.0, 1.0, 0.0])

    # Create a simplified pad by offsetting the patch
    inner_verts = patch_mesh.vertices.copy()
    outer_verts = inner_verts + normal * thickness_mm

    n_verts = len(inner_verts)
    all_verts = np.vstack([inner_verts, outer_verts])

    # Build faces: original + offset + boundary walls
    inner_faces = patch_mesh.faces.copy()
    outer_faces = inner_faces[:, ::-1] + n_verts  # flipped for outward normal
    all_faces = np.vstack([inner_faces, outer_faces])

    # Boundary walls
    edges = patch_mesh.edges_unique
    for e in edges:
        v0, v1 = e[0], e[1]
        all_faces = np.vstack([all_faces, [
            [v0, v1, v1 + n_verts],
            [v0, v1 + n_verts, v0 + n_verts],
        ]])

    pad = trimesh.Trimesh(vertices=all_verts, faces=np.array(all_faces), process=True)
    pad.remove_unreferenced_vertices()

    ts = uuid.uuid4().hex[:8]
    out_dir = OUTPUTS / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"contact_patch_{ts}.stl"
    pad.export(out_path)

    return {
        "case_id": case_id,
        "center": center_pt.tolist(),
        "radius_mm": radius_mm,
        "thickness_mm": thickness_mm,
        "clearance_mm": clearance_mm,
        "patch_faces": len(face_indices),
        "patch_vertices": len(vert_indices),
        "url": f"/api/cases/{case_id}/outputs/contact_patch_{ts}.stl",
        "info": mesh_info(pad),
    }


# ══════════════════════════════════════════════════════════════════════
# Fase 5 — Drill / Screw Trajectory Editor
# ══════════════════════════════════════════════════════════════════════

def _screw_traj_path(case_id: str) -> Path:
    return UPLOADS / case_id / "screw_trajectories.json"


def _load_screw_trajs(case_id: str) -> list:
    p = _screw_traj_path(case_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []


def _save_screw_trajs(case_id: str, data: list):
    UPLOADS.mkdir(parents=True, exist_ok=True)
    (UPLOADS / case_id).mkdir(parents=True, exist_ok=True)
    _screw_traj_path(case_id).write_text(json.dumps(data, indent=2), encoding="utf-8")


@app.get("/api/cases/{case_id}/screw-trajectories")
def get_screw_trajectories(case_id: str):
    """Get all screw trajectories for a case."""
    return {"trajectories": _load_screw_trajs(case_id)}


@app.post("/api/cases/{case_id}/screw-trajectories")
def add_screw_trajectory(
    case_id: str,
    entry_x: float = Form(...),
    entry_y: float = Form(...),
    entry_z: float = Form(...),
    dir_x: float = Form(0.0),
    dir_y: float = Form(0.0),
    dir_z: float = Form(1.0),
    length_mm: float = Form(30.0),
    diameter_mm: float = Form(3.5),
    name: str = Form(""),
):
    """Add a screw trajectory defined by entry point + direction."""
    entry = np.array([entry_x, entry_y, entry_z], dtype=float)
    direction = np.array([dir_x, dir_y, dir_z], dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise HTTPException(400, "direction vector cannot be zero")
    direction = direction / norm

    trajs = _load_screw_trajs(case_id)
    ts_id = uuid.uuid4().hex[:8]
    traj = {
        "id": ts_id,
        "name": name or f"screw_{len(trajs) + 1}",
        "entry": entry.tolist(),
        "direction": direction.tolist(),
        "length_mm": length_mm,
        "diameter_mm": diameter_mm,
        "tip": (entry + direction * length_mm).tolist(),
    }
    trajs.append(traj)
    _save_screw_trajs(case_id, trajs)

    # Generate visual cylinder for the trajectory
    in_path = UPLOADS / case_id / "input.stl"
    collision_info = None
    if in_path.exists():
        mesh = load_mesh(in_path)
        # Check intersection with bone
        tip = entry + direction * length_mm
        # Simple collision: check if entry and tip are near mesh vertices
        tree = cKDTree(mesh.vertices)
        entry_dist, _ = tree.query(entry, k=1)
        tip_dist, _ = tree.query(tip, k=1)
        collision_info = {
            "entry_distance_to_bone_mm": round(float(entry_dist), 2),
            "tip_distance_to_bone_mm": round(float(tip_dist), 2),
        }

    return {"trajectory": traj, "collision": collision_info}


@app.delete("/api/cases/{case_id}/screw-trajectories/{traj_id}")
def delete_screw_trajectory(case_id: str, traj_id: str):
    """Remove a screw trajectory by ID."""
    trajs = _load_screw_trajs(case_id)
    trajs = [t for t in trajs if t["id"] != traj_id]
    _save_screw_trajs(case_id, trajs)
    return {"ok": True, "remaining": len(trajs)}


@app.post("/api/cases/{case_id}/screw-trajectories/generate-sleeves")
def generate_screw_sleeves(
    case_id: str,
    sleeve_outer_mm: float = Form(6.0),
    sleeve_length_mm: float = Form(12.0),
):
    """Generate drill guide sleeves for all screw trajectories."""
    trajs = _load_screw_trajs(case_id)
    if not trajs:
        raise HTTPException(400, "no screw trajectories defined")

    parts = []
    for traj in trajs:
        entry = np.array(traj["entry"])
        direction = np.array(traj["direction"])
        # Create a cylinder along the trajectory
        cyl = trimesh.creation.cylinder(
            radius=sleeve_outer_mm / 2.0,
            height=sleeve_length_mm,
            sections=24,
        )
        # Orient cylinder along direction
        z_axis = np.array([0.0, 0.0, 1.0])
        if not np.allclose(direction, z_axis):
            rot_axis = np.cross(z_axis, direction)
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            angle = math.acos(np.clip(np.dot(z_axis, direction), -1, 1))
            rot_mat = trimesh.transformations.rotation_matrix(angle, rot_axis)
            cyl.apply_transform(rot_mat)
        # Position at entry point
        cyl.apply_translation(entry + direction * sleeve_length_mm / 2.0)
        parts.append(cyl)

    if parts:
        combined = trimesh.util.concatenate(parts)
        combined.remove_unreferenced_vertices()
        out_dir = OUTPUTS / case_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "screw_sleeves.stl"
        combined.export(out_path)
        return {
            "url": f"/api/cases/{case_id}/outputs/screw_sleeves.stl",
            "count": len(trajs),
            "info": mesh_info(combined),
        }
    raise HTTPException(400, "no sleeves generated")


# ══════════════════════════════════════════════════════════════════════
# Fase 6 — Patient-Specific Plate Generator
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/plate-generator")
def generate_plate(
    case_id: str,
    centerline_points: str = Form(...),  # JSON array of [x,y,z] points
    width_mm: float = Form(12.0),
    thickness_mm: float = Form(3.0),
    screw_hole_diameter_mm: float = Form(3.5),
    screw_hole_spacing_mm: float = Form(15.0),
    edge_offset_mm: float = Form(3.0),
):
    """Generate a patient-specific bone plate from a centerline path.

    The plate follows the bone surface anatomy with evenly spaced screw holes.
    """
    in_path = UPLOADS / case_id / "input.stl"
    if not in_path.exists():
        raise HTTPException(404, "case not found")

    try:
        pts = json.loads(centerline_points)
        pts = [np.array(p, dtype=float) for p in pts]
    except Exception:
        raise HTTPException(400, "invalid centerline_points: expected JSON array of [x,y,z]")

    if len(pts) < 2:
        raise HTTPException(400, "at least 2 centerline points required")

    mesh = load_mesh(in_path)
    tree = cKDTree(mesh.vertices)

    # Generate plate body by sweeping a rectangle along the centerline
    # Sample points along the path
    path_points = []
    for i in range(len(pts) - 1):
        seg_len = np.linalg.norm(pts[i + 1] - pts[i])
        n_steps = max(2, int(seg_len / 2.0))  # every 2mm
        for t in np.linspace(0, 1, n_steps):
            path_points.append(pts[i] * (1 - t) + pts[i + 1] * t)

    if not path_points:
        path_points = pts

    # For each path point, find the bone surface normal and offset
    plate_parts = []
    screw_positions = []

    for i, pp in enumerate(path_points):
        # Find nearest bone surface point
        dist, nearest_idx = tree.query(pp, k=1)
        surface_pt = mesh.vertices[nearest_idx]
        # Estimate normal from mesh face normals near this point
        face_idx = None
        for fi, face in enumerate(mesh.faces):
            if nearest_idx in face:
                face_idx = fi
                break
        if face_idx is not None:
            normal = mesh.face_normals[face_idx]
        else:
            normal = np.array([0.0, 1.0, 0.0])

        # Offset from bone surface
        offset_pt = surface_pt + normal * 1.5  # 1.5mm offset from bone

        # Create a small box segment at this position
        if i < len(path_points) - 1:
            tangent = path_points[i + 1] - pp
        elif i > 0:
            tangent = pp - path_points[i - 1]
        else:
            tangent = np.array([1.0, 0.0, 0.0])

        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-12:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = tangent / tangent_norm

        # Build local coordinate system
        binormal = np.cross(normal, tangent)
        bn_norm = np.linalg.norm(binormal)
        if bn_norm < 1e-12:
            binormal = np.cross(normal, np.array([1.0, 0.0, 0.0]))
            bn_norm = np.linalg.norm(binormal)
        binormal = binormal / bn_norm

        # Create box segment
        half_w = width_mm / 2.0
        seg_len = screw_hole_spacing_mm

        box = trimesh.creation.box(extents=[seg_len, thickness_mm, width_mm])
        # Orient: x along tangent, y along normal, z along binormal
        rot_mat = np.eye(4)
        rot_mat[:3, 0] = tangent
        rot_mat[:3, 1] = normal
        rot_mat[:3, 2] = binormal
        box.apply_transform(rot_mat)
        box.apply_translation(offset_pt)
        plate_parts.append(box)

        # Screw hole positions (every N mm)
        if i % max(1, int(screw_hole_spacing_mm / 2.0)) == 0:
            screw_positions.append(offset_pt.tolist())

    if plate_parts:
        plate = trimesh.util.concatenate(plate_parts)
        plate.remove_unreferenced_vertices()

        # Subtract screw holes (cylinders)
        hole_cyls = []
        for sp in screw_positions:
            cyl = trimesh.creation.cylinder(
                radius=screw_hole_diameter_mm / 2.0,
                height=thickness_mm * 3,
                sections=16,
            )
            cyl.apply_translation(sp)
            hole_cyls.append(cyl)

        if hole_cyls:
            try:
                holes_combined = trimesh.util.concatenate(hole_cyls)
                plate = plate.difference(holes_combined)
            except Exception:
                pass  # Keep plate without holes if boolean fails

        out_dir = OUTPUTS / case_id
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = uuid.uuid4().hex[:8]
        out_path = out_dir / f"patient_plate_{ts}.stl"
        plate.export(out_path)

        return {
            "case_id": case_id,
            "width_mm": width_mm,
            "thickness_mm": thickness_mm,
            "screw_holes": len(screw_positions),
            "screw_positions": screw_positions,
            "url": f"/api/cases/{case_id}/outputs/patient_plate_{ts}.stl",
            "info": mesh_info(plate),
        }

    raise HTTPException(400, "plate generation failed")


# ══════════════════════════════════════════════════════════════════════
# Fase 8 — DICOM Anonymization
# ══════════════════════════════════════════════════════════════════════

DICOM_TAGS_TO_ANONYMIZE = [
    "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
    "PatientAge", "PatientWeight", "PatientAddress", "PatientPhone",
    "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName",
    "InstitutionName", "InstitutionAddress", "StationName",
    "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate",
    "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime",
    "AccessionNumber", "StudyID", "OtherPatientIDs",
    "OtherPatientNames", "PhysiciansOfRecord", "RequestingPhysician",
]


@app.post("/api/cases/{case_id}/anonymize-dicom")
def anonymize_dicom(
    case_id: str,
    new_patient_id: str = Form("ANON"),
    new_patient_name: str = Form("Anonymous"),
    remove_dates: bool = Form(True),
    remove_physicians: bool = Form(True),
):
    """Anonymize DICOM files in a case. Creates anonymized copies."""
    dicom_dir = UPLOADS / case_id / "dicom"
    if not dicom_dir.exists():
        raise HTTPException(404, "case has no DICOM folder")

    anon_dir = UPLOADS / case_id / "dicom_anonymized"
    if anon_dir.exists():
        shutil.rmtree(anon_dir)
    anon_dir.mkdir(parents=True)

    count = 0
    for p in dicom_dir.rglob("*"):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(p), force=True)
            if not hasattr(ds, "PixelData"):
                continue

            # Anonymize tags
            for tag_name in DICOM_TAGS_TO_ANONYMIZE:
                if hasattr(ds, tag_name):
                    if tag_name in ("PatientName",):
                        setattr(ds, tag_name, new_patient_name)
                    elif tag_name in ("PatientID",):
                        setattr(ds, tag_name, new_patient_id)
                    elif remove_dates and "Date" in tag_name:
                        setattr(ds, tag_name, "")
                    elif remove_dates and "Time" in tag_name:
                        setattr(ds, tag_name, "")
                    elif remove_physicians and "Physician" in tag_name:
                        setattr(ds, tag_name, "")
                    elif remove_physicians and tag_name in ("OperatorsName", "PerformingPhysicianName"):
                        setattr(ds, tag_name, "")
                    else:
                        setattr(ds, tag_name, "")

            # Save anonymized
            rel = p.relative_to(dicom_dir)
            out_p = (anon_dir / rel).resolve()
            if not str(out_p).startswith(str(anon_dir.resolve())):
                continue
            out_p.parent.mkdir(parents=True, exist_ok=True)
            ds.save_as(str(out_p))
            count += 1
        except Exception:
            continue

    return {
        "anonymized_files": count,
        "anonymized_dir": str(anon_dir),
        "new_patient_id": new_patient_id,
        "new_patient_name": new_patient_name,
    }


# ══════════════════════════════════════════════════════════════════════
# Fase 8 — Mesh Mirror / Reflection
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/mirror-mesh")
def mirror_mesh(
    case_id: str,
    axis: str = Form("x"),  # x | y | z
    fragment: str = Form(""),
):
    """Mirror a mesh (or the input mesh) across a given axis."""
    in_path = UPLOADS / case_id / "input.stl"
    if fragment:
        src_path = OUTPUTS / case_id / f"fragment_{fragment}.stl"
        if not src_path.exists():
            src_path = OUTPUTS / case_id / f"{fragment}.stl"
    else:
        src_path = in_path

    if not src_path.exists():
        raise HTTPException(404, "mesh not found")

    mesh = load_mesh(src_path)
    # Mirror by negating the appropriate coordinate
    mirror_mat = np.eye(4)
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis.lower(), 0)
    mirror_mat[axis_idx, axis_idx] = -1.0
    mesh.apply_transform(mirror_mat)
    # Flip normals after mirror
    mesh.invert()

    out_dir = OUTPUTS / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = uuid.uuid4().hex[:8]
    name = fragment or "input"
    out_path = out_dir / f"mirrored_{name}_{axis}_{ts}.stl"
    mesh.export(out_path)

    return {
        "case_id": case_id,
        "axis": axis,
        "source": str(src_path.name),
        "url": f"/api/cases/{case_id}/outputs/{out_path.name}",
        "info": mesh_info(mesh),
    }


# ══════════════════════════════════════════════════════════════════════
# Fase 8 — Cephalometry (Angular Measurements)
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/cephalometry")
def cephalometry(
    case_id: str,
    landmarks: str = Form(...),  # JSON: { "S": [x,y,z], "N": [x,y,z], "A": [x,y,z], "B": [x,y,z], "Go": [x,y,z], "Me": [x,y,z] }
):
    """Compute cephalometric angles from named landmarks.

    Standard landmarks: S (Sella), N (Nasion), A (Subspinale),
    B (Supramentale), Go (Gonion), Me (Menton).

    Returns SNA, SNB, ANB, FMA, IMPA, S-N angles.
    """
    try:
        lm = json.loads(landmarks)
    except Exception:
        raise HTTPException(400, "invalid landmarks JSON")

    def get_pt(name):
        if name not in lm:
            raise HTTPException(400, f"missing landmark: {name}")
        return np.array(lm[name], dtype=float)

    results = {}

    # Helper: angle at vertex between two arms
    def angle_at(vertex, arm_a, arm_b):
        va = arm_a - vertex
        vb = arm_b - vertex
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-12 or nb < 1e-12:
            return None
        cos_a = np.clip(np.dot(va, vb) / (na * nb), -1, 1)
        return round(math.degrees(math.acos(cos_a)), 2)

    # Helper: distance
    def dist(a, b):
        return round(float(np.linalg.norm(b - a)), 2)

    try:
        S, N, A, B = get_pt("S"), get_pt("N"), get_pt("A"), get_pt("B")

        # SNA angle: S-N-A (at N)
        sna = angle_at(N, S, A)
        if sna is not None:
            results["SNA_deg"] = sna

        # SNB angle: S-N-B (at N)
        snb = angle_at(N, S, B)
        if snb is not None:
            results["SNB_deg"] = snb

        # ANB angle: A-N-B (at N)
        anb = angle_at(N, A, B)
        if anb is not None:
            results["ANB_deg"] = anb

        # S-N length
        results["S_N_mm"] = dist(S, N)

        # A-B length
        results["A_B_mm"] = dist(A, B)

        # N-A length
        results["N_A_mm"] = dist(N, A)

        # N-B length
        results["N_B_mm"] = dist(N, B)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"cephalometry calculation error: {e}")

    # Additional landmarks (Go, Me) if provided
    if "Go" in lm and "Me" in lm:
        Go = get_pt("Go")
        Me = get_pt("Me")
        results["Go_Me_mm"] = dist(Go, Me)

        if "S" in lm:
            S = get_pt("S")
            # FMA: Frankfort-Mandibular plane angle (S-N vs Go-Me)
            sn_dir = N - S
            gm_dir = Me - Go
            na, nb = np.linalg.norm(sn_dir), np.linalg.norm(gm_dir)
            if na > 1e-12 and nb > 1e-12:
                cos_a = np.clip(np.dot(sn_dir, gm_dir) / (na * nb), -1, 1)
                results["FMA_deg"] = round(math.degrees(math.acos(cos_a)), 2)

    return {"case_id": case_id, "measurements": results, "landmarks_used": list(lm.keys())}


# ══════════════════════════════════════════════════════════════════════
# Fase 8 — Case Management (CRUD, versioning, audit)
# ══════════════════════════════════════════════════════════════════════

CASE_META_DIR = DATA / "case_meta"
CASE_META_DIR.mkdir(parents=True, exist_ok=True)


def _case_meta_path(case_id: str) -> Path:
    return CASE_META_DIR / f"{case_id}.json"


def _load_case_meta(case_id: str) -> dict:
    p = _case_meta_path(case_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {
        "case_id": case_id,
        "created_at": None,
        "updated_at": None,
        "patient_id": "",
        "description": "",
        "status": "draft",
        "version": 1,
        "audit_log": [],
    }


def _save_case_meta(case_id: str, meta: dict):
    _case_meta_path(case_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _audit(case_id: str, action: str, details: str = ""):
    meta = _load_case_meta(case_id)
    if not meta["created_at"]:
        meta["created_at"] = time.time()
    meta["updated_at"] = time.time()
    meta.setdefault("audit_log", []).append({
        "timestamp": time.time(),
        "action": action,
        "details": details,
        "version": meta.get("version", 1),
    })
    # Keep last 200 audit entries
    if len(meta["audit_log"]) > 200:
        meta["audit_log"] = meta["audit_log"][-200:]
    _save_case_meta(case_id, meta)


@app.get("/api/cases/{case_id}/meta")
def get_case_meta(case_id: str):
    """Get case metadata including audit log."""
    meta = _load_case_meta(case_id)
    return meta


@app.put("/api/cases/{case_id}/meta")
def update_case_meta(
    case_id: str,
    patient_id: str = Form(""),
    description: str = Form(""),
    status: str = Form(""),
):
    """Update case metadata."""
    meta = _load_case_meta(case_id)
    if patient_id:
        meta["patient_id"] = patient_id
    if description:
        meta["description"] = description
    if status:
        meta["status"] = status
    meta["version"] = meta.get("version", 1) + 1
    _audit(case_id, "meta_update", f"status={status}")
    _save_case_meta(case_id, meta)
    return meta


@app.get("/api/cases-list")
def list_cases(limit: int = 50, status: str = ""):
    """List all cases with metadata."""
    cases = []
    for p in sorted(CASE_META_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            if status and meta.get("status") != status:
                continue
            cases.append(meta)
        except Exception:
            continue
        if len(cases) >= limit:
            break
    return {"cases": cases, "total": len(cases)}


@app.post("/api/cases/{case_id}/snapshot")
def create_snapshot(
    case_id: str,
    label: str = Form(""),
):
    """Create a named snapshot of the current case state."""
    in_path = UPLOADS / case_id / "input.stl"
    if not in_path.exists():
        raise HTTPException(404, "case not found")

    meta = _load_case_meta(case_id)
    snap_id = uuid.uuid4().hex[:8]
    snap_dir = OUTPUTS / case_id / "snapshots" / snap_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Copy input mesh
    shutil.copy2(in_path, snap_dir / "input.stl")

    # Copy all outputs
    out_dir = OUTPUTS / case_id
    if out_dir.exists():
        for f in out_dir.iterdir():
            if f.is_file() and f.suffix in (".stl", ".json", ".ply", ".obj"):
                shutil.copy2(f, snap_dir / f.name)

    # Record snapshot
    meta.setdefault("snapshots", []).append({
        "id": snap_id,
        "label": label or f"snap_{len(meta.get('snapshots', [])) + 1}",
        "created_at": time.time(),
        "path": str(snap_dir),
    })
    _audit(case_id, "snapshot_created", f"snap_id={snap_id}, label={label}")
    _save_case_meta(case_id, meta)

    return {"snapshot_id": snap_id, "label": label, "path": str(snap_dir)}


@app.get("/api/cases/{case_id}/snapshots")
def list_snapshots(case_id: str):
    """List all snapshots for a case."""
    meta = _load_case_meta(case_id)
    return {"snapshots": meta.get("snapshots", [])}


# ══════════════════════════════════════════════════════════════════════
# Fase 8 — PDF Surgical Plan Report
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/cases/{case_id}/report")
def generate_report(case_id: str):
    """Generate a JSON surgical plan report (PDF-ready structure)."""
    meta = _load_case_meta(case_id)
    in_path = UPLOADS / case_id / "input.stl"

    report = {
        "case_id": case_id,
        "generated_at": time.time(),
        "patient_id": meta.get("patient_id", ""),
        "description": meta.get("description", ""),
        "status": meta.get("status", "draft"),
        "version": meta.get("version", 1),
        "mesh_info": None,
        "dicom_info": None,
        "fragments": {},
        "guides": {},
        "screw_trajectories": _load_screw_trajs(case_id),
        "measurements": {},
        "audit_log": meta.get("audit_log", [])[-20:],  # last 20 entries
    }

    # Mesh info
    if in_path.exists():
        try:
            mesh = load_mesh(in_path)
            report["mesh_info"] = mesh_info(mesh)
        except Exception:
            pass

    # DICOM info
    dicom_meta_path = UPLOADS / case_id / "dicom_meta.json"
    if dicom_meta_path.exists():
        try:
            report["dicom_info"] = json.loads(dicom_meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Fragment transforms
    ft = _load_fragment_transforms(case_id)
    report["fragments"] = ft.get("fragments", {})

    # Guides
    out_dir = OUTPUTS / case_id
    if out_dir.exists():
        for guide_file in ["cutting_guide.stl", "conformal_cutting_guide.stl", "patient_plate.stl"]:
            if (out_dir / guide_file).exists():
                report["guides"][guide_file] = f"/api/cases/{case_id}/outputs/{guide_file}"

    return report


app.mount("/static", StaticFiles(directory=STATIC), name="static")