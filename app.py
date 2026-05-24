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

    valley_search = (bin_centers >= valley_lo) & (bin_centers <= valley_hi)
    threshold_min = -1.0  # sentinel for Otsu fallback

    if valley_search.any():
        search_hist = smoothed.copy()
        search_hist[~valley_search] = np.inf
        valley_idx = np.argmin(search_hist)
        valley_val = float(bin_centers[valley_idx])
        # Valley should be between soft tissue and bone peaks
        if st_peak_hu < valley_val < bone_peak_hu:
            threshold_min = valley_val

    # --- Otsu fallback ---
    if threshold_min < 0:
        try:
            from skimage.filters import threshold_otsu
            # Apply Otsu on the range -100..1500 HU (broader for small animals)
            bone_relevant = relevant[(relevant > -100) & (relevant < 1500)]
            if len(bone_relevant) > 100:
                threshold_min = float(threshold_otsu(bone_relevant))
                warnings.append(f"Valley detection failed; Otsu fallback threshold={threshold_min:.1f}")
            else:
                threshold_min = 150.0
                warnings.append("Valley/Otsu failed; using default 150")
        except Exception:
            threshold_min = 150.0
            warnings.append("Valley/Otsu failed; using default 150")

    # Sensitivity-based floor: don't go too low even with high sensitivity
    min_floor = {"low": 200.0, "medium": 150.0, "high": 100.0}[sensitivity]
    threshold_min = max(min_floor, min(threshold_min, 500.0))

    # --- threshold_max: 99.9th percentile of bone voxels ---
    bone_voxels = relevant[relevant > threshold_min]
    if len(bone_voxels) > 100:
        threshold_max = float(np.percentile(bone_voxels, 99.9))
        threshold_max = max(threshold_max, 800.0)  # at least 800 HU
        threshold_max = min(threshold_max, 3500.0)
    else:
        threshold_max = 2500.0

    # --- Bone voxel check ---
    bone_pct = (len(bone_voxels) / total_voxels) * 100.0
    if bone_pct < 0.5:
        warnings.append(
            f"Bone-like voxels are only {bone_pct:.2f}% of volume — "
            f"try lower threshold or check CT protocol"
        )

    return threshold_min, threshold_max, warnings


def _validate_slice_quality(mask: np.ndarray, spacing: tuple[float, float, float],
                             min_bone_fraction: float = 0.001) -> tuple[np.ndarray, int, int, list[str]]:
    """Validate per-slice bone content and auto-crop Z range.

    For each slice along axis 0 (Z), checks the fraction of bone pixels.
    Slices with bone fraction < min_bone_fraction (default 0.1%) are considered
    outside the bone region and cropped away.

    Returns (cropped_mask, z_start, z_end, warnings).
    """
    warnings: list[str] = []
    nz = mask.shape[0]
    if nz < 3:
        return mask, 0, nz, warnings

    bone_fractions = np.zeros(nz, dtype=np.float64)
    for z in range(nz):
        sl = mask[z]
        total = sl.shape[0] * sl.shape[1]
        if total > 0:
            bone_fractions[z] = np.count_nonzero(sl > 0.5) / total

    # Find first and last slices with meaningful bone content
    has_bone = bone_fractions >= min_bone_fraction
    bone_indices = np.where(has_bone)[0]

    if len(bone_indices) == 0:
        # No slices pass quality check — return original with warning
        warnings.append("No slices passed bone quality check; using full Z range")
        return mask, 0, nz, warnings

    z_start = int(bone_indices[0])
    z_end = int(bone_indices[-1]) + 1  # exclusive

    # Add small padding (up to 3 slices each side)
    pad = 3
    z_start = max(0, z_start - pad)
    z_end = min(nz, z_end + pad)

    cropped = mask[z_start:z_end].copy()
    return cropped, z_start, z_end, warnings


def _per_slice_opening(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Apply 2D binary opening to each slice independently to remove tiny noise."""
    if radius <= 0:
        return mask
    from scipy.ndimage import binary_opening as bop
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    result = np.zeros_like(mask, dtype=np.float32)
    for z in range(mask.shape[0]):
        sl = mask[z]
        if np.any(sl > 0.5):
            opened = bop(sl > 0.5, structure=struct)
            result[z] = opened.astype(np.float32)
        else:
            result[z] = sl
    return result


def _validate_mesh(mesh: trimesh.Trimesh, vol_shape: tuple, spacing: tuple) -> dict:
    """Validate generated mesh quality and return diagnostic info."""
    info = {
        "faces": len(mesh.faces),
        "is_watertight": bool(mesh.is_watertight),
        "bounds_mm": mesh.bounds.tolist(),
        "extents_mm": mesh.extents.tolist(),
        "warnings": [],
    }

    # Check if mesh is too small relative to volume
    vol_extent = max(
        vol_shape[0] * spacing[0],
        vol_shape[1] * spacing[1],
        vol_shape[2] * spacing[2],
    )
    mesh_extent = max(mesh.extents) if len(mesh.vertices) > 0 else 0

    if mesh_extent > 0 and vol_extent > 0:
        ratio = mesh_extent / vol_extent
        if ratio > 0.95:
            info["warnings"].append("Mesh covers almost entire volume — threshold may be too low")
        elif ratio < 0.05:
            info["warnings"].append("Mesh is very small relative to volume — threshold may be too high")

    if len(mesh.faces) < 100:
        info["warnings"].append("Very few faces generated — check HU threshold")

    if len(mesh.faces) > 500000:
        info["warnings"].append("Very dense mesh — consider increasing step_size or decimation")

    return info


def _apply_morphology(mask: np.ndarray, closing_radius: int = 0, opening_radius: int = 0,
                       dilation_radius: int = 0, erosion_radius: int = 0,
                       fill_holes: bool = False) -> np.ndarray:
    """Apply binary morphology operations to a 3D mask.

    Order: close → open → dilate → erode → fill holes.
    """
    result = mask.astype(np.float32)
    if closing_radius > 0:
        struct_close = np.ones((2 * closing_radius + 1,) * 3, dtype=np.uint8)
        closed = binary_closing(result, structure=struct_close)
        result = closed.astype(np.float32)
    if opening_radius > 0:
        struct_open = np.ones((2 * opening_radius + 1,) * 3, dtype=np.uint8)
        opened = binary_opening(result, structure=struct_open)
        result = opened.astype(np.float32)
    if dilation_radius > 0:
        struct_dil = np.ones((2 * dilation_radius + 1,) * 3, dtype=np.uint8)
        dilated = binary_dilation(result, structure=struct_dil)
        result = dilated.astype(np.float32)
    if erosion_radius > 0:
        struct_erode = np.ones((2 * erosion_radius + 1,) * 3, dtype=np.uint8)
        eroded = binary_erosion(result, structure=struct_erode)
        result = eroded.astype(np.float32)
    if fill_holes:
        from scipy.ndimage import binary_fill_holes as bfh
        filled = bfh(result > 0.5)
        result = filled.astype(np.float32)
    return result


def _remove_small_islands(mask: np.ndarray, spacing: tuple[float, float, float],
                           min_volume_mm3: float = 500.0) -> np.ndarray:
    """Remove connected components below a minimum physical volume."""
    if min_volume_mm3 <= 0:
        return mask
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
    if voxel_volume <= 0:
        return mask
    labels, num = ndi_label(mask > 0.5)
    if num == 0:
        return mask
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0  # background
    min_voxels = max(1, int(min_volume_mm3 / voxel_volume))
    keep = np.where(component_sizes >= min_voxels)[0]
    if len(keep) == 0:
        return np.zeros_like(mask)
    result = np.zeros_like(mask, dtype=np.float32)
    for k in keep:
        result[labels == k] = 1.0
    return result


def _roi_crop(mask: np.ndarray, spacing: tuple[float, float, float],
              pad_voxels: int = 3) -> tuple[np.ndarray, tuple[int, int, int], tuple[int, int, int]]:
    """Auto-crop mask to bounding box around bone voxels with padding.

    Returns cropped mask, origin offset in voxels (z0,y0,x0), and original shape.
    """
    orig_shape = mask.shape
    bone_voxels = np.argwhere(mask > 0.5)
    if len(bone_voxels) == 0:
        return mask, (0, 0, 0), orig_shape
    mins = bone_voxels.min(axis=0)
    maxs = bone_voxels.max(axis=0)
    mins = np.maximum(mins - pad_voxels, 0)
    maxs = np.minimum(maxs + pad_voxels + 1, np.array(mask.shape))
    z0, y0, x0 = mins
    z1, y1, x1 = maxs
    cropped = mask[z0:z1, y0:y1, x0:x1].copy()
    return cropped, (int(z0), int(y0), int(x0)), orig_shape


def _pca_align_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align bone vertices so that principal axes map onto CAD axes.

    Returns aligned vertices and the 4x4 transform matrix applied.
    """
    if len(vertices) < 4:
        return vertices, np.eye(4)
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    cov = np.cov(centered.T)
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        return vertices, np.eye(4)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by descending eigenvalue
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    # Rotation that maps the principal axes to x,y,z
    rot = eigenvectors.T
    # Ensure right-handed coordinate system
    if np.linalg.det(rot) < 0:
        rot[2] *= -1
    aligned = centered @ rot.T + centroid
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = centroid - rot @ centroid
    return aligned, transform


def _cleanup_mesh(mesh: trimesh.Trimesh, smoothing_iterations: int = 0,
                   decimation_target_faces: Optional[int] = None,
                   repair: bool = True,
                   smoothing_method: str = "taubin",
                   preserve_volume: bool = True) -> trimesh.Trimesh:
    """Post-process mesh: smooth, decimate, repair normals/watertight.

    smoothing_method:
      "taubin" — Taubin (lambda-mu) smoothing, preserves features better than Laplacian
      "laplacian" — standard Laplacian smoothing
      "hc" — HC Laplacian (Humphrey's Class), preserves volume best
      "none" — skip smoothing

    Taubin smoothing is recommended for bone meshes: it reduces the "shrinkage"
    effect of Laplacian smoothing and preserves joint surface detail better.
    """
    out = mesh.copy()

    if smoothing_iterations > 0 and smoothing_method != "none":
        try:
            if smoothing_method == "taubin":
                # Taubin smoothing: alternating shrink/grow to preserve volume
                lambda_val = 0.5
                mu_val = -0.53
                for _ in range(int(smoothing_iterations)):
                    trimesh.smoothing.filter_humphrey(out, lamb=lambda_val, iterations=1)
            elif smoothing_method == "hc":
                # HC Laplacian with volume preservation
                trimesh.smoothing.filter_humphrey(out, lamb=0.5, iterations=int(smoothing_iterations))
            else:
                # Standard Laplacian (original behavior)
                trimesh.smoothing.filter_laplacian(
                    out, lamb=0.35, iterations=int(smoothing_iterations)
                )
        except Exception:
            pass

    if decimation_target_faces is not None and decimation_target_faces > 0 and len(out.faces) > decimation_target_faces:
        try:
            out = out.simplify_quadric_decimation(face_count=decimation_target_faces)
        except Exception:
            pass

    if repair:
        try:
            trimesh.repair.fix_winding(out)
            trimesh.repair.fix_normals(out)
            trimesh.repair.fill_holes(out)
            trimesh.repair.fix_inversion(out)
        except Exception:
            pass

    out.remove_unreferenced_vertices()
    return out


def _vtk_windowed_sinc_smooth(mesh: trimesh.Trimesh, iterations: int = 15,
                                band: float = 0.1, pass_band: float = 0.1,
                                feature_angle: float = 120.0,
                                edge_angle: float = 90.0,
                                boundary_smoothing: bool = True,
                                feature_smoothing: bool = False,
                                non_manifold_smoothing: bool = True,
                                normalize_coordinates: bool = True) -> trimesh.Trimesh:
    """VTK Windowed Sinc smoothing — same as 3D Slicer / Horos.

    This is the key filter that makes Slicer bone renderings look professional.
    Unlike Laplacian smoothing, Windowed Sinc preserves sharp features and
    edge detail while removing noise. It works in the frequency domain.

    Parameters match VTK's vtkWindowedSincPolyDataFilter defaults used by Slicer:
      iterations: number of smoothing passes (15-30 typical for bone meshes)
      band: windowed sinc filter band (0.1 = good balance)
      pass_band: pass band value (0.1 = preserves detail, lower = smoother)
      feature_angle: angle for feature edge detection (120° default)
      edge_angle: angle for smoothing along edges (90° default)
      feature_smoothing: if True, smooths along sharp edges too (off = preserve edges)
      boundary_smoothing: if True, smooths boundary vertices
      non_manifold_smoothing: if True, smooths non-manifold vertices
    """
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    except ImportError:
        # Fallback to trimesh Taubin if VTK not available
        mesh_smooth = mesh.copy()
        try:
            for _ in range(iterations):
                trimesh.smoothing.filter_humphrey(mesh_smooth, lamb=0.5, iterations=1)
        except Exception:
            pass
        return mesh_smooth

    try:
        # Convert trimesh → VTK polydata
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(mesh.vertices.astype(np.float64)))

        polys = vtk.vtkCellArray()
        for face in mesh.faces:
            polys.InsertNextCell(3)
            polys.InsertCellPoint(int(face[0]))
            polys.InsertCellPoint(int(face[1]))
            polys.InsertCellPoint(int(face[2]))

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polys)

        # Compute normals first (required for feature detection)
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(polydata)
        normals_filter.ConsistencyOn()
        normals_filter.AutoOrientNormalsOn()
        normals_filter.Update()
        input_data = normals_filter.GetOutput()

        # Windowed Sinc smoothing — the Slicer/Horos algorithm
        smooth_filter = vtk.vtkWindowedSincPolyDataFilter()
        smooth_filter.SetInputData(input_data)
        smooth_filter.SetNumberOfIterations(int(iterations))
        smooth_filter.SetPassBand(float(pass_band))
        smooth_filter.SetFeatureAngle(float(feature_angle))
        smooth_filter.SetEdgeAngle(float(edge_angle))
        smooth_filter.SetBoundarySmoothing(boundary_smoothing)
        smooth_filter.SetFeatureEdgeSmoothing(feature_smoothing)
        smooth_filter.SetNonManifoldSmoothing(non_manifold_smoothing)
        smooth_filter.SetNormalizeCoordinates(normalize_coordinates)
        smooth_filter.Update()

        result = smooth_filter.GetOutput()

        # Convert VTK → trimesh
        out_verts = vtk_to_numpy(result.GetPoints().GetData()).astype(np.float64)
        out_faces_list = []
        for i in range(result.GetNumberOfPolys()):
            cell = result.GetCell(i)
            ids = cell.GetPointIds()
            out_faces_list.append([ids.GetId(0), ids.GetId(1), ids.GetId(2)])
        out_faces = np.array(out_faces_list, dtype=np.int64)

        return trimesh.Trimesh(vertices=out_verts, faces=out_faces, process=True)
    except Exception:
        return mesh


def _split_mesh_into_bones(mesh: trimesh.Trimesh, min_volume_mm3: float = 50.0) -> list:
    """Split a mesh into separate bone meshes using connected components.

    This mimics 3D Slicer's 'Split Islands' function.
    Each connected component above min_volume_mm3 becomes a separate bone mesh.

    Returns list of (bone_name, bone_mesh) tuples.
    """
    try:
        # Use trimesh built-in split
        meshes = mesh.split(only_watertight=False)
    except Exception:
        return [("bone_0", mesh)]

    if len(meshes) <= 1:
        return [("bone_0", mesh)]

    # Filter by minimum volume and sort by volume (largest first)
    bone_list = []
    for i, m in enumerate(meshes):
        try:
            vol = m.volume if m.is_watertight else m.bounding_box_oriented.volume
        except Exception:
            vol = 0.0
        if vol >= min_volume_mm3:
            bone_list.append((f"bone_{i}", m, vol))

    # Sort by volume descending
    bone_list.sort(key=lambda x: x[2], reverse=True)

    # Rename with anatomical hints based on position and size
    result = []
    for idx, (name, bm, vol) in enumerate(bone_list):
        # Try to guess bone name from position
        centroid = bm.centroid
        z_pos = centroid[2] if len(centroid) > 2 else 0
        bone_name = f"bone_{idx}"
        result.append((bone_name, bm))

    return result if result else [("bone_0", mesh)]


def apply_preset(params: dict, preset_name: Optional[str],
                  custom_hu_min: Optional[float] = None,
                  custom_hu_max: Optional[float] = None) -> dict:
    """Merge a segmentation preset with any user overrides."""
    if preset_name and preset_name.lower() in SEGMENTATION_PRESETS:
        merged = dict(SEGMENTATION_PRESETS[preset_name.lower()])
    else:
        merged = dict(SEGMENTATION_PRESETS["long_bone"])
    if custom_hu_min is not None:
        merged["threshold_hu"] = custom_hu_min
    if custom_hu_max is not None:
        merged["threshold_max_hu"] = custom_hu_max
    merged["_active_preset"] = preset_name or "long_bone"
    return merged


def dicom_to_bone_mesh(
    dicom_dir: Path,
    threshold_hu: float = 250.0,
    step_size: int = 1,
    series_uid: Optional[str] = None,
    threshold_max_hu: Optional[float] = None,
    keep_largest: bool = True,
    smoothing_iterations: int = 2,
    # --- Segmentation Engine v3 parameters ---
    morphology_closing_radius: int = 0,
    morphology_opening_radius: int = 0,
    dilation_radius: int = 0,
    erosion_radius: int = 0,
    fill_holes: bool = False,
    min_island_volume_mm3: float = 0.0,
    roi_crop: bool = True,
    pca_align: bool = False,
    decimation_target_faces: Optional[int] = None,
    repair_mesh: bool = True,
    preset: Optional[str] = None,
    case_dir: Optional[Path] = None,
    # --- v3 new ---
    auto_threshold_enabled: bool = True,
    per_slice_denoise: bool = True,
    z_crop: bool = True,
    sensitivity: str = "medium",
    smoothing_method: str = "taubin",
) -> tuple[trimesh.Trimesh, dict]:
    """Convert DICOM CT series to bone mesh with Segmentation Engine v3.

    New v3 pipeline:
      Load DICOM → auto-threshold (histogram + Otsu fallback)
      → HU mask → per-slice denoise → Z crop to bone region
      → 3D morphological closing → 3D morphological opening
      → remove small islands → keep largest component
      → fill holes → marching cubes → smooth → decimate → repair → [PCA align]
    """
    seg_warnings: list[str] = []

    # Apply preset defaults then override with explicit params
    if preset and preset.lower() in SEGMENTATION_PRESETS:
        preset_cfg = SEGMENTATION_PRESETS[preset.lower()]
        if auto_threshold_enabled and threshold_hu == 250.0:
            threshold_hu = preset_cfg["threshold_hu"]  # will be overridden by auto
        if threshold_max_hu is None:
            threshold_max_hu = preset_cfg.get("threshold_max_hu")
        if morphology_closing_radius == 0:
            morphology_closing_radius = preset_cfg.get("morphology_closing_radius", 0)
        if morphology_opening_radius == 0:
            morphology_opening_radius = preset_cfg.get("morphology_opening_radius", 0)
        if not fill_holes:
            fill_holes = preset_cfg.get("fill_holes", False)
        if min_island_volume_mm3 <= 0:
            min_island_volume_mm3 = preset_cfg.get("min_island_volume_mm3", 0.0)
        if smoothing_iterations == 2:
            smoothing_iterations = preset_cfg.get("smoothing_iterations", 2)
        if decimation_target_faces is None:
            decimation_target_faces = preset_cfg.get("decimation_target_faces", None)
        active_preset = preset.lower()
    else:
        active_preset = None

    vol, spacing, meta = load_dicom_series(dicom_dir, selected_series_uid=series_uid)
    hu_min_actual = float(np.min(vol))
    hu_max_actual = float(np.max(vol))

    # Cache volume as .npy for fast MPR slice access
    if case_dir is not None:
        npy_path = case_dir / "volume_cache.npy"
        if not npy_path.exists():
            np.save(str(npy_path), vol)

    # --- Auto-threshold: adaptive HU threshold from histogram analysis ---
    auto_info: dict = {}
    if auto_threshold_enabled:
        auto_tmin, auto_tmax, auto_warnings = _auto_threshold(vol, sensitivity=sensitivity)
        # Only override if user didn't explicitly set a custom threshold
        if threshold_hu == 250.0 or (preset and preset.lower() in SEGMENTATION_PRESETS):
            threshold_hu = auto_tmin
        if threshold_max_hu is None or threshold_max_hu == 3000.0:
            threshold_max_hu = auto_tmax
        seg_warnings.extend(auto_warnings)
        auto_info = {
            "auto_threshold_min": auto_tmin,
            "auto_threshold_max": auto_tmax,
            "auto_warnings": auto_warnings,
        }

    # Validate threshold against actual volume range
    if threshold_hu <= hu_min_actual or threshold_hu >= hu_max_actual:
        raise ValueError(f"threshold {threshold_hu} outside volume HU range {hu_min_actual}..{hu_max_actual}")

    # --- HU mask ---
    if threshold_max_hu is not None and threshold_max_hu > threshold_hu:
        mask = (vol >= threshold_hu) & (vol <= threshold_max_hu)
        src = mask.astype(np.float32)
    else:
        # Raw HU threshold mode (original behavior, no binary mask)
        src = None

    crop_offset = (0, 0, 0)
    z_crop_range = (0, vol.shape[0])

    if src is not None:
        # --- Per-slice denoise: 2D opening on each slice to remove tiny noise ---
        if per_slice_denoise and morphology_opening_radius > 0:
            src = _per_slice_opening(src, radius=morphology_opening_radius)

        # --- Z crop: auto-crop to slices containing bone ---
        if z_crop:
            src, z_start, z_end, z_warnings = _validate_slice_quality(src, spacing)
            z_crop_range = (z_start, z_end)
            seg_warnings.extend(z_warnings)

        # --- ROI crop (bounding box around bone voxels) ---
        if roi_crop:
            src, crop_offset, orig_shape = _roi_crop(src, spacing)

        # --- 3D Morphology: closing → opening ---
        src = _apply_morphology(
            src,
            closing_radius=morphology_closing_radius,
            opening_radius=morphology_opening_radius if not per_slice_denoise else 0,
            dilation_radius=dilation_radius,
            erosion_radius=erosion_radius,
            fill_holes=False,  # fill holes after island removal
        )

        # --- Remove small islands ---
        if min_island_volume_mm3 > 0:
            src = _remove_small_islands(src, spacing, min_volume_mm3=min_island_volume_mm3)

        # --- Keep largest connected component ---
        if keep_largest:
            labels = measure.label(src > 0.5, connectivity=1)
            if labels.max() > 0:
                counts = np.bincount(labels.ravel())
                counts[0] = 0
                largest_label = int(counts.argmax())
                largest_size = counts[largest_label]
                total_labeled = counts.sum()
                # Only keep largest if it's a meaningful fraction (>30% of labeled voxels)
                if total_labeled > 0 and largest_size / total_labeled > 0.3:
                    src = (labels == largest_label).astype(np.float32)
                else:
                    seg_warnings.append(
                        f"Largest component only {largest_size/total_labeled:.0%} of labeled voxels — keeping all"
                    )

        # --- Fill holes ---
        if fill_holes:
            src = _apply_morphology(src, fill_holes=True)

    # --- Marching cubes ---
    if src is not None:
        verts, faces, normals, values = measure.marching_cubes(
            src,
            level=0.5,
            spacing=spacing,
            step_size=max(1, int(step_size)),
            allow_degenerate=False,
        )
    else:
        # Raw HU threshold mode (original behavior)
        verts, faces, normals, values = measure.marching_cubes(
            vol,
            level=threshold_hu,
            spacing=spacing,
            step_size=max(1, int(step_size)),
            allow_degenerate=False,
        )

    # skimage returns z,y,x coordinates. Convert to x,y,z for CAD-like display.
    verts_xyz = verts[:, [2, 1, 0]]

    # Adjust for ROI crop offset in physical space
    if roi_crop and crop_offset != (0, 0, 0):
        offset_phys = np.array([
            crop_offset[0] * spacing[0],
            crop_offset[1] * spacing[1],
            crop_offset[2] * spacing[2],
        ])
        verts_xyz += offset_phys

    mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()

    # --- PCA alignment ---
    pca_transform = None
    if pca_align and len(mesh.vertices) >= 4:
        aligned_verts, pca_transform = _pca_align_vertices(mesh.vertices.copy())
        mesh = trimesh.Trimesh(vertices=aligned_verts, faces=mesh.faces.copy(), process=True)
        mesh.remove_unreferenced_vertices()

    # --- Mesh cleanup (smooth, decimate, repair) ---
    mesh = _cleanup_mesh(
        mesh,
        smoothing_iterations=smoothing_iterations,
        decimation_target_faces=decimation_target_faces,
        repair=repair_mesh,
        smoothing_method=smoothing_method,
    )

    manifold_report = {
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent) if len(mesh.faces) > 0 else None,
        "euler_number": int(mesh.euler_number) if len(mesh.faces) > 0 else None,
        "bounds_mm": mesh.bounds.tolist(),
        "extents_mm": mesh.extents.tolist(),
    }

    meta.update({
        "threshold_hu": threshold_hu,
        "threshold_max_hu": threshold_max_hu,
        "keep_largest": bool(keep_largest),
        "smoothing_iterations": int(smoothing_iterations),
        "step_size": int(step_size),
        "morphology_closing_radius": morphology_closing_radius,
        "morphology_opening_radius": morphology_opening_radius,
        "dilation_radius": dilation_radius,
        "erosion_radius": erosion_radius,
        "fill_holes": fill_holes,
        "min_island_volume_mm3": min_island_volume_mm3,
        "roi_crop": roi_crop,
        "roi_crop_offset_voxels": list(crop_offset),
        "z_crop_range": list(z_crop_range),
        "pca_align": pca_align,
        "pca_transform": pca_transform.tolist() if pca_transform is not None else None,
        "decimation_target_faces": decimation_target_faces,
        "repair_mesh": repair_mesh,
        "active_preset": active_preset,
        "auto_threshold": auto_info,
        "per_slice_denoise": per_slice_denoise,
        "generated_vertices": int(len(mesh.vertices)),
        "generated_faces": int(len(mesh.faces)),
        "mesh_validation": manifold_report,
        "segmentation_warnings": seg_warnings,
        "warning": "Segmentation Engine v3. Needs manual validation for surgical use.",
    })
    return mesh, meta


def dicom_to_bone_meshes(
    dicom_dir: Path,
    threshold_hu: float = 150.0,
    step_size: int = 1,
    series_uid: Optional[str] = None,
    threshold_max_hu: Optional[float] = None,
    smoothing_iterations: int = 30,
    smoothing_method: str = "vtk_sinc",
    min_bone_volume_mm3: float = 50.0,
    preset: Optional[str] = None,
    case_dir: Optional[Path] = None,
    sensitivity: str = "medium",
    max_bones: int = 20,
    separation_erosion: int = 2,
) -> tuple[list, dict]:
    """Convert DICOM CT series to MULTIPLE bone meshes using watershed segmentation.

    WATERSHED APPROACH (standard in medical imaging, used by 3D Slicer):
      1. Load DICOM → auto-threshold → binary mask
      2. Compute distance transform (how far each voxel is from the surface)
      3. Find markers via heavy erosion (erosion_radius=5-8 voxels)
      4. Watershed on negative distance transform with markers
         → naturally cuts bones at the thinnest points between them
      5. For each watershed region: extract mask → marching cubes → smooth
      6. VTK Windowed Sinc smoothing per bone (pass_band=0.01 for very smooth)

    Returns (list of (bone_name, mesh), meta_dict).
    """
    from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
    from scipy.ndimage import label as ndi_label
    from scipy.ndimage import distance_transform_edt
    from skimage import measure
    from skimage.segmentation import watershed as sk_watershed
    from skimage.feature import peak_local_max

    # Load volume
    vol, spacing, meta = load_dicom_series(dicom_dir, selected_series_uid=series_uid)
    hu_min_actual = float(np.min(vol))
    hu_max_actual = float(np.max(vol))

    # Cache volume
    if case_dir is not None:
        npy_path = case_dir / "volume_cache.npy"
        if not npy_path.exists():
            np.save(str(npy_path), vol)

    # Auto-threshold
    auto_info = {}
    if sensitivity:
        auto_tmin, auto_tmax, auto_warnings = _auto_threshold(vol, sensitivity=sensitivity)
        if threshold_hu == 150.0:
            threshold_hu = auto_tmin
        if threshold_max_hu is None or threshold_max_hu == 3000.0:
            threshold_max_hu = auto_tmax
        auto_info = {"auto_threshold_min": auto_tmin, "auto_threshold_max": auto_tmax}

    # Create binary mask
    if threshold_max_hu is not None and threshold_max_hu > threshold_hu:
        mask = (vol >= threshold_hu) & (vol <= threshold_max_hu)
    else:
        mask = vol >= threshold_hu
    src = mask.astype(np.float32)

    # Z crop to bone region
    _, z_start, z_end, _ = _validate_slice_quality(src, spacing)
    src = src[z_start:z_end]

    # ROI crop
    src, crop_offset, orig_shape = _roi_crop(src, spacing)

    # Fill holes in the mask (important for cortical bone)
    for z in range(src.shape[0]):
        slice_mask = src[z] > 0.5
        if slice_mask.any():
            src[z] = binary_fill_holes(slice_mask).astype(np.float32)

        # ========== CONNECTED COMPONENTS BONE SEPARATION ==========
    # Simple approach: find connected components in the binary mask.
    # Each connected component = one bone. No watershed needed.
    # Bones are naturally separated by soft tissue in CT.

    print(f"[bones] Finding connected components on {src.shape} volume...")

    # Light morphological closing to fill small gaps within bones
    from scipy.ndimage import binary_closing, generate_binary_structure, gaussian_filter
    struct_close = generate_binary_structure(3, 1)  # 6-connectivity
    src_binary = binary_closing(src > 0.5, structure=struct_close, iterations=2)

    # Find connected components
    labels, num_components = ndi_label(src_binary)
    print(f"[bones] Found {num_components} connected components")

    # Extract volumes
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
    bone_list = []

    for bone_id in range(1, num_components + 1):
        bone_mask = (labels == bone_id).astype(np.float32)
        bone_volume = float(bone_mask.sum()) * voxel_volume

        if bone_volume < min_bone_volume_mm3:
            print(f"[bones] Skipping component {bone_id}: volume {bone_volume:.0f} mm³ < {min_bone_volume_mm3}")
            continue

        # Fill holes in this specific bone
        for z in range(bone_mask.shape[0]):
            sl = bone_mask[z] > 0.5
            if sl.any():
                bone_mask[z] = binary_fill_holes(sl).astype(np.float32)

        # Remove small islands
        bone_mask = _remove_small_islands(bone_mask, spacing, min_volume_mm3=min_bone_volume_mm3 * 0.5)

        if bone_mask.sum() < 10:
            continue

        # Marching cubes for this bone
        try:
            verts, faces, normals, values = measure.marching_cubes(
                bone_mask,
                level=0.5,
                spacing=spacing,
                step_size=max(1, int(step_size)),
                allow_degenerate=False,
            )
        except Exception as e:
            print(f"[bones] Marching cubes failed for component {bone_id}: {e}")
            continue

        # Convert to x,y,z
        verts_xyz = verts[:, [2, 1, 0]]

        # Adjust for crop offset
        if crop_offset != (0, 0, 0):
            offset_phys = np.array([
                crop_offset[0] * spacing[0],
                crop_offset[1] * spacing[1],
                crop_offset[2] * spacing[2],
            ])
            verts_xyz += offset_phys

        bone_mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=True)
        bone_mesh.remove_unreferenced_vertices()

        # Basic repair
        try:
            trimesh.repair.fix_winding(bone_mesh)
            trimesh.repair.fix_normals(bone_mesh)
            trimesh.repair.fill_holes(bone_mesh)
        except Exception:
            pass

        # VTK Windowed Sinc smoothing — pass_band=0.01 for very smooth surface
        if smoothing_method == "vtk_sinc" and smoothing_iterations > 0:
            bone_mesh = _vtk_windowed_sinc_smooth(
                bone_mesh,
                iterations=smoothing_iterations,
                pass_band=0.01,
                feature_angle=120.0,
                feature_smoothing=False,
                boundary_smoothing=True,
                non_manifold_smoothing=True,
                normalize_coordinates=True,
            )

        print(f"[bones] Component {bone_id}: {len(bone_mesh.vertices)} verts, {len(bone_mesh.faces)} faces, vol={bone_volume:.0f} mm³")
        bone_list.append((f"bone_{len(bone_list)}", bone_mesh, bone_volume))

    # Sort by volume descending
    bone_list.sort(key=lambda x: x[2], reverse=True)
    bone_list = bone_list[:max_bones]
    num_bones = len(bone_list)
    print(f"[bones] Final: {num_bones} bones extracted")
    # Build result

    smoothed_bones = [(name, bm) for name, bm, _ in bone_list]

    meta["bone_count"] = len(smoothed_bones)
    meta["auto_threshold"] = auto_info
    meta["separation_method"] = "connected_components"
    
    meta["bones"] = []
    for name, bm in smoothed_bones:
        try:
            vol_val = bm.volume if bm.is_watertight else bm.bounding_box_oriented.volume
        except Exception:
            vol_val = 0.0
        meta["bones"].append({
            "name": name,
            "vertices": len(bm.vertices),
            "faces": len(bm.faces),
            "volume_mm3": round(vol_val, 2),
            "centroid": bm.centroid.tolist(),
            "bounds": bm.bounds.tolist(),
        })

    return smoothed_bones, meta


@app.post("/api/upload-dicom-chunk")
async def upload_dicom_chunk(
    files: list[UploadFile] = File(...),
    chunk_index: int = Form(0),
    total_chunks: int = Form(1),
    upload_id: str = Form(""),
    split_bones: bool = Form(False),
    sensitivity: str = Form("medium"),
):
    """Receive DICOM files in chunks. First chunk creates case, last chunk triggers segmentation."""
    if not upload_id:
        raise HTTPException(400, "upload_id required")

    case_dir = UPLOADS / upload_id
    dicom_dir = case_dir / "dicom"
    dicom_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for i, up in enumerate(files):
        raw_name = (up.filename or f"slice_{chunk_index}_{i:05d}.dcm").replace("\\", "/")
        parts = [p for p in raw_name.split("/") if p and p not in {".", ".."}]
        rel = Path(*parts) if parts else Path(f"slice_{chunk_index}_{i:05d}.dcm")
        target = (dicom_dir / rel).resolve()
        if not str(target).startswith(str(dicom_dir.resolve())):
            raise ValueError("unsafe folder path")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            shutil.copyfileobj(up.file, f)
        written += 1

    result = {
        "upload_id": upload_id,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "files_written": written,
        "status": "chunk_received",
    }

    # Last chunk: trigger segmentation
    if chunk_index >= total_chunks - 1:
        try:
            if split_bones:
                bones, meta = dicom_to_bone_meshes(
                    dicom_dir, sensitivity=sensitivity or "medium",
                    case_dir=case_dir,
                )
            else:
                mesh, meta = dicom_to_bone_mesh(dicom_dir, sensitivity=sensitivity or "medium")
            bones_dir = case_dir / "bones"
            bones_dir.mkdir(parents=True, exist_ok=True)
            bone_metas = []
            combined = None
            for i, (bone_name, bone_mesh) in enumerate(bones):
                bone_path = bones_dir / f"bone_{i}.stl"
                bone_mesh.export(bone_path)
                if combined is None:
                    combined = bone_mesh.copy()
                else:
                    combined = trimesh.util.concatenate([combined, bone_mesh])
                bmeta = meta["bones"][i] if i < len(meta.get("bones", [])) else {}
                bmeta["mesh_url"] = f"/api/cases/{upload_id}/bone/{i}.stl"
                bone_metas.append(bmeta)
            if combined is not None:
                combined.export(case_dir / "input.stl")
            (case_dir / "dicom_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            result["status"] = "complete"
            result["case_id"] = upload_id
            result["bone_count"] = len(bones)
            result["bones"] = bone_metas
            result["dicom"] = meta
        except Exception as e:
            result["status"] = "segmentation_error"
            result["error"] = str(e)

    return result

@app.get("/")
def root():
    return FileResponse(
        STATIC / "index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/health")
def health():
    return {"ok": True, "app": "3DMedicalPlanner", "version": "0.1.0"}


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "mesh.stl").suffix.lower()
    if ext not in {".stl", ".obj", ".ply"}:
        raise HTTPException(400, "Upload STL/OBJ/PLY for MVP v0.1")
    case_id = uuid.uuid4().hex[:10]
    case_dir = UPLOADS / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    in_path = case_dir / f"input{ext}"
    with in_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        mesh = load_mesh(in_path)
    except Exception as e:
        raise HTTPException(400, f"Mesh invalid: {e}")
    mesh.export(case_dir / "input.stl")
    return {"case_id": case_id, "mesh_url": f"/api/cases/{case_id}/input.stl", "info": mesh_info(mesh)}


@app.post("/api/sample")
def sample():
    src = make_sample_bone()
    case_id = uuid.uuid4().hex[:10]
    case_dir = UPLOADS / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    dst = case_dir / "input.stl"
    shutil.copy2(src, dst)
    mesh = load_mesh(dst)
    return {"case_id": case_id, "mesh_url": f"/api/cases/{case_id}/input.stl", "info": mesh_info(mesh)}


@app.post("/api/upload-dicom")
def upload_dicom(
    file: UploadFile = File(...),
    threshold_hu: float = Form(250.0),
    step_size: int = Form(1),
    # --- Segmentation Engine v2 ---
    threshold_max_hu: float = Form(3000.0),
    keep_largest: bool = Form(True),
    smoothing_iterations: int = Form(2),
    morphology_closing_radius: int = Form(0),
    morphology_opening_radius: int = Form(0),
    dilation_radius: int = Form(0),
    erosion_radius: int = Form(0),
    fill_holes: bool = Form(False),
    min_island_volume_mm3: float = Form(0.0),
    roi_crop: bool = Form(True),
    pca_align: bool = Form(False),
    decimation_target_faces: int = Form(0),
    repair_mesh: bool = Form(True),
    preset: str = Form(""),
    sensitivity: str = Form("medium"),
    split_bones: bool = Form(False),
):
    ext = Path(file.filename or "dicom.zip").suffix.lower()
    if ext not in {".zip", ".dcm", ""}:
        raise HTTPException(400, "Upload DICOM .zip or .dcm series file")
    case_id = uuid.uuid4().hex[:10]
    case_dir = UPLOADS / case_id
    dicom_dir = case_dir / "dicom"
    dicom_dir.mkdir(parents=True, exist_ok=True)
    raw_path = case_dir / (file.filename or "dicom_upload")
    with raw_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        if ext == ".zip":
            with zipfile.ZipFile(raw_path) as z:
                for member in z.infolist():
                    target = (dicom_dir / member.filename).resolve()
                    if not str(target).startswith(str(dicom_dir.resolve())):
                        raise ValueError("unsafe zip path")
                z.extractall(dicom_dir)
        else:
            shutil.copy2(raw_path, dicom_dir / "slice.dcm")
        if split_bones:
            bones, meta = dicom_to_bone_meshes(
                dicom_dir,
                threshold_hu=threshold_hu,
                step_size=step_size,
                series_uid=None,
                threshold_max_hu=threshold_max_hu if threshold_max_hu > threshold_hu else None,
                smoothing_iterations=smoothing_iterations,
                preset=preset or None,
                case_dir=case_dir,
                sensitivity=sensitivity or "medium",
            )
        else:
            mesh, meta = dicom_to_bone_mesh(
                dicom_dir,
                threshold_hu=threshold_hu,
                step_size=step_size,
                threshold_max_hu=threshold_max_hu if threshold_max_hu > threshold_hu else None,
                keep_largest=keep_largest,
                smoothing_iterations=smoothing_iterations,
                morphology_closing_radius=morphology_closing_radius,
                morphology_opening_radius=morphology_opening_radius,
                dilation_radius=dilation_radius,
                erosion_radius=erosion_radius,
                fill_holes=fill_holes,
                min_island_volume_mm3=min_island_volume_mm3,
                roi_crop=roi_crop,
                pca_align=pca_align,
                decimation_target_faces=decimation_target_faces if decimation_target_faces > 0 else None,
                repair_mesh=repair_mesh,
                preset=preset or None,
                case_dir=case_dir,
                auto_threshold_enabled=True,
                per_slice_denoise=True,
                z_crop=True,
                sensitivity=sensitivity or "medium",
            )
    except Exception as e:
        shutil.rmtree(case_dir, ignore_errors=True)
        raise HTTPException(400, f"DICOM conversion failed: {e}")

    if split_bones:
        # Save each bone as a separate STL
        bones_dir = case_dir / "bones"
        bones_dir.mkdir(parents=True, exist_ok=True)
        bone_metas = []
        combined = None
        for i, (bone_name, bone_mesh) in enumerate(bones):
            bone_path = bones_dir / f"bone_{i}.stl"
            bone_mesh.export(bone_path)
            if combined is None:
                combined = bone_mesh.copy()
            else:
                combined = trimesh.util.concatenate([combined, bone_mesh])
            bmeta = meta["bones"][i] if i < len(meta.get("bones", [])) else {}
            bmeta["mesh_url"] = f"/api/cases/{case_id}/bone/{i}.stl"
            bone_metas.append(bmeta)
        # Save combined mesh for backward compatibility
        if combined is not None:
            combined.export(case_dir / "input.stl")
        (case_dir / "dicom_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {
            "case_id": case_id,
            "mesh_url": f"/api/cases/{case_id}/input.stl",
            "bone_count": len(bones),
            "bones": bone_metas,
            "dicom": meta,
        }
    else:
        out_path = case_dir / "input.stl"
        mesh.export(out_path)
        (case_dir / "dicom_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {
            "case_id": case_id,
            "mesh_url": f"/api/cases/{case_id}/input.stl",
            "info": mesh_info(mesh),
            "dicom": meta,
        }


@app.post("/api/upload-dicom-folder")
def upload_dicom_folder(
    files: list[UploadFile] = File(...),
    threshold_hu: float = Form(250.0),
    step_size: int = Form(1),
    # --- Segmentation Engine v2 ---
    threshold_max_hu: float = Form(3000.0),
    keep_largest: bool = Form(True),
    smoothing_iterations: int = Form(2),
    morphology_closing_radius: int = Form(0),
    morphology_opening_radius: int = Form(0),
    dilation_radius: int = Form(0),
    erosion_radius: int = Form(0),
    fill_holes: bool = Form(False),
    min_island_volume_mm3: float = Form(0.0),
    roi_crop: bool = Form(True),
    pca_align: bool = Form(False),
    decimation_target_faces: int = Form(0),
    repair_mesh: bool = Form(True),
    preset: str = Form(""),
    sensitivity: str = Form("medium"),
    split_bones: bool = Form(False),
):
    if not files:
        raise HTTPException(400, "No DICOM files received")
    case_id = uuid.uuid4().hex[:10]
    case_dir = UPLOADS / case_id
    dicom_dir = case_dir / "dicom"
    dicom_dir.mkdir(parents=True, exist_ok=True)
    try:
        written = 0
        for i, up in enumerate(files):
            # Browser folder upload may send relative path in filename. Preserve subfolders safely.
            raw_name = (up.filename or f"slice_{i:05d}.dcm").replace("\\", "/")
            parts = [p for p in raw_name.split("/") if p and p not in {".", ".."}]
            rel = Path(*parts) if parts else Path(f"slice_{i:05d}.dcm")
            target = (dicom_dir / rel).resolve()
            if not str(target).startswith(str(dicom_dir.resolve())):
                raise ValueError("unsafe folder path")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as f:
                shutil.copyfileobj(up.file, f)
            written += 1
        if split_bones:
            bones, meta = dicom_to_bone_meshes(
                dicom_dir,
                threshold_hu=threshold_hu,
                step_size=step_size,
                series_uid=None,
                threshold_max_hu=threshold_max_hu if threshold_max_hu > threshold_hu else None,
                smoothing_iterations=smoothing_iterations,
                preset=preset or None,
                case_dir=case_dir,
                sensitivity=sensitivity or "medium",
            )
        else:
            mesh, meta = dicom_to_bone_mesh(
                dicom_dir,
                threshold_hu=threshold_hu,
                step_size=step_size,
                threshold_max_hu=threshold_max_hu if threshold_max_hu > threshold_hu else None,
                keep_largest=keep_largest,
                smoothing_iterations=smoothing_iterations,
                morphology_closing_radius=morphology_closing_radius,
                morphology_opening_radius=morphology_opening_radius,
                dilation_radius=dilation_radius,
                erosion_radius=erosion_radius,
                fill_holes=fill_holes,
                min_island_volume_mm3=min_island_volume_mm3,
                roi_crop=roi_crop,
                pca_align=pca_align,
                decimation_target_faces=decimation_target_faces if decimation_target_faces > 0 else None,
                repair_mesh=repair_mesh,
                preset=preset or None,
                case_dir=case_dir,
                auto_threshold_enabled=True,
                per_slice_denoise=True,
                z_crop=True,
                sensitivity=sensitivity or "medium",
            )
    except Exception as e:
        shutil.rmtree(case_dir, ignore_errors=True)
        raise HTTPException(400, f"DICOM folder conversion failed: {e}")

    if split_bones:
        # Save each bone as a separate STL
        bones_dir = case_dir / "bones"
        bones_dir.mkdir(parents=True, exist_ok=True)
        bone_metas = []
        combined = None
        for i, (bone_name, bone_mesh) in enumerate(bones):
            bone_path = bones_dir / f"bone_{i}.stl"
            bone_mesh.export(bone_path)
            if combined is None:
                combined = bone_mesh.copy()
            else:
                combined = trimesh.util.concatenate([combined, bone_mesh])
            bmeta = meta["bones"][i] if i < len(meta.get("bones", [])) else {}
            bmeta["mesh_url"] = f"/api/cases/{case_id}/bone/{i}.stl"
            bone_metas.append(bmeta)
        # Save combined mesh for backward compatibility
        if combined is not None:
            combined.export(case_dir / "input.stl")
        meta["uploaded_files"] = written
        (case_dir / "dicom_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {
            "case_id": case_id,
            "mesh_url": f"/api/cases/{case_id}/input.stl",
            "bone_count": len(bones),
            "bones": bone_metas,
            "dicom": meta,
        }
    else:
        out_path = case_dir / "input.stl"
        mesh.export(out_path)
        meta["uploaded_files"] = written
        (case_dir / "dicom_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {
            "case_id": case_id,
            "mesh_url": f"/api/cases/{case_id}/input.stl",
            "info": mesh_info(mesh),
            "dicom": meta,
        }


@app.get("/api/recent-cases")
def recent_cases(limit: int = 10):
    rows = []
    for d in sorted([p for p in UPLOADS.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        inp = d / "input.stl"
        if not inp.exists():
            continue
        meta_path = d / "dicom_meta.json"
        meta = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = None
        rows.append({
            "case_id": d.name,
            "mesh_url": f"/api/cases/{d.name}/input.stl",
            "size_bytes": inp.stat().st_size,
            "is_dicom": meta is not None,
            "dicom": meta,
        })
    return rows


@app.get("/api/cases/{case_id}/dicom-info")
def dicom_info(case_id: str):
    dicom_dir = UPLOADS / case_id / "dicom"
    if not dicom_dir.exists():
        raise HTTPException(404, "case has no DICOM folder")
    try:
        _vol, _spacing, meta = load_dicom_series(dicom_dir)
        return meta
    except Exception as e:
        raise HTTPException(400, f"DICOM info failed: {e}")


@app.post("/api/cases/{case_id}/regenerate-mesh")
def regenerate_mesh(
    case_id: str,
    threshold_hu: float = Form(250.0),
    threshold_max_hu: float = Form(3000.0),
    step_size: int = Form(1),
    series_uid: str = Form(""),
    keep_largest: bool = Form(True),
    smoothing_iterations: int = Form(2),
    # --- Segmentation Engine v2 ---
    morphology_closing_radius: int = Form(0),
    morphology_opening_radius: int = Form(0),
    dilation_radius: int = Form(0),
    erosion_radius: int = Form(0),
    fill_holes: bool = Form(False),
    min_island_volume_mm3: float = Form(0.0),
    roi_crop: bool = Form(True),
    pca_align: bool = Form(False),
    decimation_target_faces: int = Form(0),
    repair_mesh: bool = Form(True),
    preset: str = Form(""),
):
    case_dir = UPLOADS / case_id
    dicom_dir = case_dir / "dicom"
    if not dicom_dir.exists():
        raise HTTPException(404, "case has no DICOM folder")

    # Push undo snapshot before regenerating
    _push_undo(case_id, "regenerate_mesh", {"dicom_dir": str(dicom_dir)})

    try:
        mesh, meta = dicom_to_bone_mesh(
            dicom_dir,
            threshold_hu=threshold_hu,
            threshold_max_hu=threshold_max_hu if threshold_max_hu > threshold_hu else None,
            step_size=step_size,
            series_uid=series_uid or None,
            keep_largest=keep_largest,
            smoothing_iterations=smoothing_iterations,
            morphology_closing_radius=morphology_closing_radius,
            morphology_opening_radius=morphology_opening_radius,
            dilation_radius=dilation_radius,
            erosion_radius=erosion_radius,
            fill_holes=fill_holes,
            min_island_volume_mm3=min_island_volume_mm3,
            roi_crop=roi_crop,
            pca_align=pca_align,
            decimation_target_faces=decimation_target_faces if decimation_target_faces > 0 else None,
            repair_mesh=repair_mesh,
            preset=preset or None,
            case_dir=case_dir,
        )
    except Exception as e:
        raise HTTPException(400, f"Regenerate mesh failed: {e}")
    out_path = case_dir / "input.stl"
    mesh.export(out_path)
    (case_dir / "dicom_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"case_id": case_id, "mesh_url": f"/api/cases/{case_id}/input.stl", "info": mesh_info(mesh), "dicom": meta}


@app.get("/api/cases/{case_id}/input.stl")
def input_stl(case_id: str):
    path = UPLOADS / case_id / "input.stl"
    if not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="model/stl", filename=f"{case_id}-input.stl")


@app.get("/api/cases/{case_id}/bone/{bone_index}.stl")
def get_bone_stl(case_id: str, bone_index: int):
    bone_path = UPLOADS / case_id / "bones" / f"bone_{bone_index}.stl"
    if not bone_path.exists():
        raise HTTPException(404, "Bone not found")
    return FileResponse(bone_path, media_type="model/stl")


@app.get("/api/cases/{case_id}/outputs/{name}")
def output_file(case_id: str, name: str):
    if "/" in name or ".." in name:
        raise HTTPException(400, "bad name")
    path = OUTPUTS / case_id / name
    if not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="model/stl", filename=name)


@app.post("/api/cut")
def cut(
    case_id: str,
    axis: str = Form("z"),
    offset_mm: float = Form(0.0),
    rotate_side: str = Form("positive"),
    rotate_axis: str = Form("z"),
    rotate_deg: float = Form(0.0),
    translate_x: float = Form(0.0),
    translate_y: float = Form(0.0),
    translate_z: float = Form(0.0),
):
    in_path = UPLOADS / case_id / "input.stl"
    if not in_path.exists():
        raise HTTPException(404, "case not found")

    # Push undo snapshot before cutting
    _push_undo(case_id, "cut", {"axis": axis, "offset_mm": offset_mm})

    mesh = load_mesh(in_path)
    normal = axis_normal(axis)
    center = mesh.bounds.mean(axis=0)
    origin = center + normal * offset_mm

    pos = safe_slice(mesh, normal, origin)
    neg = safe_slice(mesh, -normal, origin)

    pivot = origin
    side = rotate_side.lower()
    if side == "negative":
        neg = translate_mesh(rotate_mesh(neg, rotate_axis, rotate_deg, pivot), translate_x, translate_y, translate_z)
    else:
        pos = translate_mesh(rotate_mesh(pos, rotate_axis, rotate_deg, pivot), translate_x, translate_y, translate_z)

    out_dir = OUTPUTS / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_path = out_dir / "fragment_positive.stl"
    neg_path = out_dir / "fragment_negative.stl"
    combined_path = out_dir / "bone_cut_repositioned.stl"
    meta_path = out_dir / "plan.json"

    pos.export(pos_path)
    neg.export(neg_path)
    combined = trimesh.util.concatenate([pos, neg])
    combined.export(combined_path)

    plan = {
        "case_id": case_id,
        "plane": {"axis": axis, "origin_mm": origin.tolist(), "normal": normal.tolist(), "offset_mm": offset_mm},
        "transform": {
            "rotate_side": rotate_side,
            "rotate_axis": rotate_axis,
            "rotate_deg": rotate_deg,
            "translate_mm": [translate_x, translate_y, translate_z],
        },
        "input": mesh_info(mesh),
        "positive": mesh_info(pos),
        "negative": mesh_info(neg),
        "combined": mesh_info(combined),
        "outputs": {
            "positive": f"/api/cases/{case_id}/outputs/fragment_positive.stl",
            "negative": f"/api/cases/{case_id}/outputs/fragment_negative.stl",
            "combined": f"/api/cases/{case_id}/outputs/bone_cut_repositioned.stl",
            "plan": f"/api/cases/{case_id}/outputs/plan.json",
        },
        "warning": "MVP research prototype. Mesh split uses open cuts without capping; not medical/manufacturing validated.",
    }
    meta_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return JSONResponse(plan)


@app.get("/api/cases/{case_id}/outputs/plan.json")
def plan_json(case_id: str):
    path = OUTPUTS / case_id / "plan.json"
    if not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="application/json", filename=f"{case_id}-plan.json")


@app.post("/api/guide")
def guide(
    case_id: str,
    axis: str = Form("y"),
    offset_mm: float = Form(0.0),
    length_mm: float = Form(70.0),
    height_mm: float = Form(22.0),
    rail_thickness_mm: float = Form(6.0),
    slot_width_mm: float = Form(1.2),
    pin_radius_mm: float = Form(1.6),
    pin_spacing_mm: float = Form(38.0),
    pin_sleeve_outer_mm: float = Form(3.2),
):
    in_path = UPLOADS / case_id / "input.stl"
    if not in_path.exists():
        raise HTTPException(404, "case not found")
    if length_mm <= 0 or height_mm <= 0 or rail_thickness_mm <= 0 or slot_width_mm <= 0:
        raise HTTPException(400, "guide dimensions must be positive")
    if pin_radius_mm <= 0 or pin_spacing_mm < 0:
        raise HTTPException(400, "pin parameters invalid")

    # Push undo snapshot before generating guide
    _push_undo(case_id, "guide", {"axis": axis, "offset_mm": offset_mm, "length_mm": length_mm})

    mesh = load_mesh(in_path)
    guide_mesh, guide_meta = generate_cutting_guide(
        mesh=mesh,
        axis=axis,
        offset_mm=offset_mm,
        length_mm=length_mm,
        height_mm=height_mm,
        rail_thickness_mm=rail_thickness_mm,
        slot_width_mm=slot_width_mm,
        pin_radius_mm=pin_radius_mm,
        pin_spacing_mm=pin_spacing_mm,
        pin_sleeve_outer_mm=pin_sleeve_outer_mm,
    )
    out_dir = OUTPUTS / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    guide_path = out_dir / "cutting_guide.stl"
    meta_path = out_dir / "guide.json"
    guide_mesh.export(guide_path)
    payload = {
        "case_id": case_id,
        "guide": guide_meta,
        "mesh": mesh_info(guide_mesh),
        "outputs": {
            "guide": f"/api/cases/{case_id}/outputs/cutting_guide.stl",
            "guide_json": f"/api/cases/{case_id}/outputs/guide.json",
        },
        "warning": "First-pass guide: printable parametric saw-slot/pin-sleeve jig, not yet conformal/contact-fit to patient bone.",
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return JSONResponse(payload)


@app.post("/api/conformal-guide")
def conformal_guide(
    case_id: str,
    axis: str = Form("y"),
    offset_mm: float = Form(0.0),
    contact_side: str = Form("positive"),
    length_mm: float = Form(70.0),
    height_mm: float = Form(28.0),
    thickness_mm: float = Form(4.0),
    clearance_mm: float = Form(0.6),
    resolution_u: int = Form(30),
    resolution_v: int = Form(16),
    include_slot_rails: bool = Form(True),
):
    in_path = UPLOADS / case_id / "input.stl"
    if not in_path.exists():
        raise HTTPException(404, "case not found")
    if length_mm <= 0 or height_mm <= 0 or thickness_mm <= 0:
        raise HTTPException(400, "conformal dimensions must be positive")
    if resolution_u < 3 or resolution_v < 3 or resolution_u > 80 or resolution_v > 60:
        raise HTTPException(400, "resolution out of range")

    # Push undo snapshot before generating conformal guide
    _push_undo(case_id, "conformal_guide", {"axis": axis, "offset_mm": offset_mm, "contact_side": contact_side})

    mesh = load_mesh(in_path)
    try:
        pad, pad_meta = generate_conformal_pad(
            mesh=mesh,
            axis=axis,
            offset_mm=offset_mm,
            contact_side=contact_side,
            length_mm=length_mm,
            height_mm=height_mm,
            thickness_mm=thickness_mm,
            clearance_mm=clearance_mm,
            resolution_u=resolution_u,
            resolution_v=resolution_v,
        )
    except Exception as e:
        raise HTTPException(400, f"conformal generation failed: {e}")

    output_mesh = pad
    rail_meta = None
    if include_slot_rails:
        rails, rail_meta = generate_cutting_guide(
            mesh=mesh,
            axis=axis,
            offset_mm=offset_mm,
            length_mm=length_mm,
            height_mm=max(8.0, height_mm * 0.45),
            rail_thickness_mm=4.0,
            slot_width_mm=1.2,
            pin_radius_mm=1.5,
            pin_spacing_mm=max(18.0, length_mm * 0.55),
            pin_sleeve_outer_mm=3.2,
        )
        output_mesh = trimesh.util.concatenate([pad, rails])
        output_mesh.remove_unreferenced_vertices()

    out_dir = OUTPUTS / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pad_path = out_dir / "conformal_contact_pad.stl"
    guide_path = out_dir / "conformal_cutting_guide.stl"
    meta_path = out_dir / "conformal_guide.json"
    pad.export(pad_path)
    output_mesh.export(guide_path)
    payload = {
        "case_id": case_id,
        "pad": pad_meta,
        "slot_rails": rail_meta,
        "pad_mesh": mesh_info(pad),
        "combined_mesh": mesh_info(output_mesh),
        "outputs": {
            "pad": f"/api/cases/{case_id}/outputs/conformal_contact_pad.stl",
            "combined": f"/api/cases/{case_id}/outputs/conformal_cutting_guide.stl",
            "json": f"/api/cases/{case_id}/outputs/conformal_guide.json",
        },
        "warning": "Conformal MVP: contact pad follows local mesh via sampled nearest-surface approximation. Needs validation/boolean refinement before surgical use.",
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return JSONResponse(payload)


@app.post("/api/cases/{case_id}/undo")
def undo_action(case_id: str):
    mgr = get_manager(case_id)
    if not mgr.can_undo():
        raise HTTPException(409, "nothing to undo")
    result = mgr.undo()
    if result is None:
        raise HTTPException(409, "nothing to undo")
    return {"ok": True, "undone": result["action"], "history": mgr.history()}


@app.post("/api/cases/{case_id}/redo")
def redo_action(case_id: str):
    mgr = get_manager(case_id)
    if not mgr.can_redo():
        raise HTTPException(409, "nothing to redo")
    result = mgr.redo()
    if result is None:
        raise HTTPException(409, "nothing to redo")
    return {"ok": True, "redone": result["action"], "history": mgr.history()}


@app.get("/api/cases/{case_id}/history")
def action_history(case_id: str):
    mgr = get_manager(case_id)
    return mgr.history()


def _push_undo(case_id: str, name: str, extra: Optional[dict] = None):
    """Helper: push a snapshot-based undo action before a state-changing op."""
    mgr = get_manager(case_id)
    payload = {"name": name}
    if extra:
        payload.update(extra)
    action = UndoAction(
        name=name,
        undo_payload=dict(payload),
        redo_payload=dict(payload),
    )
    mgr.push(action)


# ─── Advanced Osteotomy: helpers ───────────────────────────────────────────────

def _fragment_transforms_path(case_id: str) -> Path:
    return OUTPUTS / case_id / "fragment_transforms.json"


def _load_fragment_transforms(case_id: str) -> dict:
    p = _fragment_transforms_path(case_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"fragments": {}, "planes": [], "version": 1}


def _save_fragment_transforms(case_id: str, data: dict):
    out_dir = OUTPUTS / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _fragment_transforms_path(case_id).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _plane_from_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (origin, normal) from three points defining a plane."""
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, normal := v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise ValueError("degenerate plane: three points are collinear")
    normal = normal / norm
    origin = (p1 + p2 + p3) / 3.0
    return origin, normal


def _split_mesh_by_planes(
    mesh: trimesh.Trimesh,
    planes: list[dict],
) -> list[trimesh.Trimesh]:
    """Split mesh sequentially by multiple plane-twexpr{plane} cuts.

    Each plane dict: {"origin": [x,y,z], "normal": [x,y,z]}.
    Returns list of fragment meshes.
    """
    # We assign each face to a fragment based on which side of each plane its centroid falls.
    centroids = mesh.triangles_center  # (N, 3)
    n = len(centroids)
    # Each plane splits space into positive (dot >= 0) and negative (dot < 0).
    # We build a binary signature per face, one bit per plane.
    bits = np.zeros((n, len(planes)), dtype=np.int8)
    for pi, plane in enumerate(planes):
        origin = np.array(plane["origin"])
        normal = np.array(plane["normal"])
        vals = (centroids - origin) @ normal
        bits[:, pi] = (vals >= 0).astype(np.int8)
    # Convert bit-patterns to unique group ids
    # Use binary encoding: each row of bits is a binary number
    powers = 1 << np.arange(len(planes), dtype=np.int64)
    group_ids = bits @ powers  # (N,) integer labels
    unique_groups = np.unique(group_ids)
    fragments = []
    for gid in unique_groups:
        mask = group_ids == gid
        if not mask.any():
            continue
        face_indices = np.where(mask)[0]
        # Extract sub-mesh by face selection
        if len(face_indices) == 0:
            continue
        face_set = mesh.faces[face_indices]
        used_verts = np.unique(face_set)
        if len(used_verts) == 0:
            continue
        # Remap vertex indices
        remap = {int(old): i for i, old in enumerate(used_verts)}
        new_faces = np.vectorize(remap.get)(face_set).astype(np.int64)
        new_verts = mesh.vertices[used_verts]
        sub = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=True)
        sub.remove_unreferenced_vertices()
        if len(sub.faces) > 0:
            fragments.append(sub)
    return fragments if fragments else [mesh.copy()]


def _apply_transform_to_mesh(mesh: trimesh.Trimesh, transform: dict) -> trimesh.Trimesh:
    """Apply a JSON transform dict to a mesh copy. Supports translation + axis-angle rotation."""
    out = mesh.copy()
    if not transform:
        return out
    # Translation
    t = transform.get("translation_mm", [0, 0, 0])
    if any(abs(v) > 1e-12 for v in t):
        out.apply_translation(t)
    # Rotation
    r = transform.get("rotation", None)
    if r:
        axis_vec = np.array(r.get("axis", [0, 0, 1], dtype=float))
        angle_deg = float(r.get("degrees", 0))
        pivot = np.array(r.get("pivot", [0, 0, 0], dtype=float))
        if abs(angle_deg) > 1e-9:
            mat = trimesh.transformations.rotation_matrix(
                math.radians(angle_deg), axis_vec, point=pivot
            )
            out.apply_transform(mat)
    return out


# ─── POST /api/cut-from-points ────────────────────────────────────────────────

@app.post("/api/cut-from-points")
def cut_from_points(
    case_id: str,
    point1: str = Form(...),  # JSON [x,y,z]
    point2: str = Form(...),
    point3: str = Form(...),
    fragment_name_pos: str = Form("positive"),  # named fragment on normal+ side
    fragment_name_neg: str = Form("negative"),
):
    """Define a cutting plane from 3 points clicked on the mesh."""
    in_path = UPLOADS / case_id / "input.stl"
    if not in_path.exists():
        raise HTTPException(404, "case not found")
    try:
        p1 = np.array(json.loads(point1), dtype=float)
        p2 = np.array(json.loads(point2), dtype=float)
        p3 = np.array(json.loads(point3), dtype=float)
    except Exception:
        raise HTTPException(400, "invalid points: expected JSON arrays [x,y,z]")
    if p1.shape != (3,) or p2.shape != (3,) or p3.shape != (3,):
        raise HTTPException(400, "each point must be [x,y,z]")
    try:
        origin, normal = _plane_from_points(p1, p2, p3)
    except ValueError as e:
        raise HTTPException(400, str(e))

    mesh = load_mesh(in_path)
    pos = safe_slice(mesh, normal, origin)
    neg = safe_slice(mesh, -normal, origin)

    out_dir = OUTPUTS / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = uuid.uuid4().hex[:8]
    pos_path = out_dir / f"fragment_{fragment_name_pos}_{ts}.stl"
    neg_path = out_dir / f"fragment_{fragment_name_neg}_{ts}.stl"
    pos.export(pos_path)
    neg.export(neg_path)

    # Update fragment transforms
    ft = _load_fragment_transforms(case_id)
    ft["fragments"][fragment_name_pos] = {
        "file": str(pos_path.name),
        "url": f"/api/cases/{case_id}/outputs/{pos_path.name}",
        "color": "#60a5fa",
    }
    ft["fragments"][fragment_name_neg] = {
        "file": str(neg_path.name),
        "url": f"/api/cases/{case_id}/outputs/{neg_path.name}",
        "color": "#f97316",
    }
    # Store plane definition
    ft["planes"].append({
        "id": ts,
        "origin": origin.tolist(),
        "normal": normal.tolist(),
        "points": [p1.tolist(), p2.tolist(), p3.tolist()],
    })
    _save_fragment_transforms(case_id, ft)

    return {
        "case_id": case_id,
        "plane": {"origin_mm": origin.tolist(), "normal_mm": normal.tolist()},
        "fragments": {
            fragment_name_pos: {"url": f"/api/cases/{case_id}/outputs/{pos_path.name}", "info": mesh_info(pos)},
            fragment_name_neg: {"url": f"/api/cases/{case_id}/outputs/{neg_path.name}", "info": mesh_info(neg)},
        },
        "fragment_transform_url": f"/api/cases/{case_id}/fragment-transforms",
    }


# ─── POST /api/cut-multi ──────────────────────────────────────────────────────


@app.post("/api/cut-bone")
def cut_bone(
    case_id: str = Form(...),
    bone_index: int = Form(...),
    plane_origin: str = Form(...),   # JSON [x,y,z]
    plane_normal: str = Form(...),   # JSON [x,y,z]
):
    """Cut a specific bone mesh along a plane defined by origin + normal.
    Returns two fragment meshes as vertex/face data for the frontend."""
    bone_path = UPLOADS / case_id / "bones" / f"bone_{bone_index}.stl"
    if not bone_path.exists():
        raise HTTPException(404, f"Bone {bone_index} not found for case {case_id}")

    try:
        origin = np.array(json.loads(plane_origin), dtype=float)
        normal = np.array(json.loads(plane_normal), dtype=float)
    except Exception:
        raise HTTPException(400, "Invalid plane_origin or plane_normal: expected JSON [x,y,z]")

    norm_val = np.linalg.norm(normal)
    if norm_val < 1e-12:
        raise HTTPException(400, "Plane normal is degenerate (zero vector)")
    normal = normal / norm_val

    mesh = load_mesh(bone_path)

    # Cut on positive side of plane
    pos_fragment = safe_slice(mesh, normal, origin, cap=True)
    # Cut on negative side of plane
    neg_fragment = safe_slice(mesh, -normal, origin, cap=True)

    # Save fragments
    out_dir = UPLOADS / case_id / "fragments"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = uuid.uuid4().hex[:8]
    pos_name = f"frag_pos_{bone_index}_{ts}.stl"
    neg_name = f"frag_neg_{bone_index}_{ts}.stl"
    pos_fragment.export(out_dir / pos_name)
    neg_fragment.export(out_dir / neg_name)

    # Return mesh data for frontend
    def mesh_to_data(m, name):
        if len(m.vertices) == 0:
            return {"name": name, "vertices": [], "faces": [], "empty": True}
        return {
            "name": name,
            "vertices": m.vertices.tolist(),
            "faces": m.faces.tolist(),
            "empty": False,
            "vertex_count": len(m.vertices),
            "face_count": len(m.faces),
        }

    return {
        "case_id": case_id,
        "bone_index": bone_index,
        "plane": {"origin": origin.tolist(), "normal": normal.tolist()},
        "positive": mesh_to_data(pos_fragment, f"pos_bone_{bone_index}"),
        "negative": mesh_to_data(neg_fragment, f"neg_bone_{bone_index}"),
    }

@app.post("/api/cut-multi")
def cut_multi(
    case_id: str,
    plane1_p1: str = Form(""),
    plane1_p2: str = Form(""),
    plane1_p3: str = Form(""),
    plane2_p1: str = Form(""),
    plane2_p2: str = Form(""),
    plane2_p3: str = Form(""),
    plane3_p1: str = Form(""),
    plane3_p2: str = Form(""),
    plane3_p3: str = Form(""),
):
    """Cut mesh with up to 3 planes at once (multi-plane osteotomy).

    Supply planeN_p1/p2/p3 as JSON [x,y,z]. Empty strings are ignored.
    Returns fragment STLs and a combined plan JSON.
    """
    in_path = UPLOADS / case_id / "input.stl"
    if not in_path.exists():
        raise HTTPException(404, "case not found")

    planes = []
    for suffix in ("1", "2", "3"):
        p1 = locals()[f"plane{suffix}_p1"]
        p2 = locals()[f"plane{suffix}_p2"]
        p3 = locals()[f"plane{suffix}_p3"]
        if p1 and p2 and p3:
            try:
                pts = [np.array(json.loads(p)) for p in (p1, p2, p3)]
            except Exception:
                raise HTTPException(400, f"plane {suffix}: invalid point JSON")
            origin, normal = _plane_from_points(*pts)
            planes.append({"origin": origin.tolist(), "normal": normal.tolist()})

    if not planes:
        raise HTTPException(400, "at least one plane (plane1) is required")

    mesh = load_mesh(in_path)
    fragments = _split_mesh_by_planes(mesh, planes)

    out_dir = OUTPUTS / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = uuid.uuid4().hex[:8]
    outputs = {}
    colors = ["#60a5fa", "#f97316", "#22c55e", "#e879f9", "#facc15", "#fb923c", "#38bdf8", "#a78bfa"]
    for i, frag in enumerate(fragments):
        name = f"fragment_{i}_{ts}"
        fpath = out_dir / f"{name}.stl"
        frag.export(fpath)
        color = colors[i % len(colors)]
        outputs[name] = {"url": f"/api/cases/{case_id}/outputs/{fpath.name}", "info": mesh_info(frag), "color": color}

    # Save transforms
    ft = _load_fragment_transforms(case_id)
    for name, info in outputs.items():
        ft["fragments"][name] = {"file": f"{name}.stl", "url": info["url"], "color": info["color"]}
    ft["planes"].extend([{**p, "id": f"{ts}_{j}"} for j, p in enumerate(planes)])
    _save_fragment_transforms(case_id, ft)

    return {
        "case_id": case_id,
        "planes": planes,
        "fragments": outputs,
        "fragment_transform_url": f"/api/cases/{case_id}/fragment-transforms",
    }


# ─── Fragment transforms CRUD ─────────────────────────────────────────────────

@app.get("/api/cases/{case_id}/fragment-transforms")
def get_fragment_transforms(case_id: str):
    """Get stored fragment transforms + plane definitions for this case."""
    ft = _load_fragment_transforms(case_id)
    return ft


@app.post("/api/cases/{case_id}/fragment-transforms/{fragment_name}")
def update_fragment_transform(
    case_id: str,
    fragment_name: str,
    translation_x: float = Form(0.0),
    translation_y: float = Form(0.0),
    translation_z: float = Form(0.0),
    rotation_axis_x: float = Form(0.0),
    rotation_axis_y: float = Form(0.0),
    rotation_axis_z: float = Form(1.0),
    rotation_deg: float = Form(0.0),
    pivot_x: float = Form(0.0),
    pivot_y: float = Form(0.0),
    pivot_z: float = Form(0.0),
):
    """Update transform for a named fragment. Re-generates STL from original cut output."""
    ft = _load_fragment_transforms(case_id)
    frag = ft.get("fragments", {}).get(fragment_name)
    if not frag or "file" not in frag:
        # Try loading from outputs
        raise HTTPException(404, f"fragment '{fragment_name}' not found")

    # Apply transform
    new_transform = {
        "translation_mm": [translation_x, translation_y, translation_z],
        "rotation": {
            "axis": [rotation_axis_x, rotation_axis_y, rotation_axis_z],
            "degrees": rotation_deg,
            "pivot": [pivot_x, pivot_y, pivot_z],
        },
    }

    # Load original fragment mesh
    src_path = OUTPUTS / case_id / frag["file"]
    if not src_path.exists():
        raise HTTPException(404, "fragment STL not found on disk")
    mesh = load_mesh(src_path)
    transformed = _apply_transform_to_mesh(mesh, new_transform)

    # Save updated STL
    out_path = OUTPUTS / case_id / f"fragment_{fragment_name}_transformed.stl"
    transformed.export(out_path)

    # Update stored transform
    frag["transform"] = new_transform
    frag["transformed_url"] = f"/api/cases/{case_id}/outputs/{out_path.name}"
    ft["fragments"][fragment_name] = frag
    _save_fragment_transforms(case_id, ft)

    return {
        "fragment": fragment_name,
        "transform": new_transform,
        "url": f"/api/cases/{case_id}/outputs/{out_path.name}",
        "info": mesh_info(transformed),
    }


# ─── POST /api/export-fragments ───────────────────────────────────────────────

@app.post("/api/export-fragments")
def export_fragments(
    case_id: str,
):
    """Export zip containing all fragment STLs + combined plan JSON."""
    ft = _load_fragment_transforms(case_id)
    out_dir = OUTPUTS / case_id
    if not out_dir.exists():
        raise HTTPException(404, "case outputs not found")

    zip_path = out_dir / "osteotomy_export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        # Write transforms JSON
        zf.writestr("fragment_transforms.json", json.dumps(ft, indent=2))
        # Include all fragment STLs
        added = set()
        for fname, finfo in ft.get("fragments", {}).items():
            src = out_dir / finfo["file"]
            if src.exists() and finfo["file"] not in added:
                zf.write(src, arcname=finfo["file"])
                added.add(finfo["file"])
            # Also include transformed version if present
            tf = finfo.get("transformed_url", "")
            if tf:
                tname = tf.split("/")[-1]
                tp = out_dir / tname
                if tp.exists() and tname not in added:
                    zf.write(tp, arcname=tname)
                    added.add(tname)

    return {"url": f"/api/cases/{case_id}/outputs/osteotomy_export.zip"}


@app.get("/api/cases/{case_id}/outputs/osteotomy_export.zip")
def download_export_zip(case_id: str):
    path = OUTPUTS / case_id / "osteotomy_export.zip"
    if not path.exists():
        raise HTTPException(404, "export not found, run export-fragments first")
    return FileResponse(path, media_type="application/zip", filename=f"{case_id}-osteotomy-export.zip")


# ─── Measurement tools ────────────────────────────────────────────────────────

@app.post("/api/measure-distance")
def measure_distance(
    case_id: str,
    point_a: str = Form(...),  # JSON [x,y,z]
    point_b: str = Form(...),
):
    """Measure Euclidean distance between two points on the mesh."""
    try:
        a = np.array(json.loads(point_a), dtype=float)
        b = np.array(json.loads(point_b), dtype=float)
    except Exception:
        raise HTTPException(400, "invalid points")
    if a.shape != (3,) or b.shape != (3,):
        raise HTTPException(400, "each point must be [x,y,z]")
    dist = float(np.linalg.norm(b - a))
    mid = ((a + b) / 2.0).tolist()
    return {
        "distance_mm": dist,
        "point_a": a.tolist(),
        "point_b": b.tolist(),
        "midpoint": mid,
    }


@app.post("/api/measure-angle")
def measure_angle(
    case_id: str,
    vertex: str = Form(...),    # JSON [x,y,z] — the vertex point
    arm_a: str = Form(...),     # JSON [x,y,z] — end of first arm
    arm_b: str = Form(...),     # JSON [x,y,z] — end of second arm
):
    """Measure angle at vertex between two arms (vertex→arm_a and vertex→arm_b)."""
    try:
        v = np.array(json.loads(vertex), dtype=float)
        a = np.array(json.loads(arm_a), dtype=float)
        b = np.array(json.loads(arm_b), dtype=float)
    except Exception:
        raise HTTPException(400, "invalid points")
    va = a - v
    vb = b - v
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        raise HTTPException(400, "degenerate arms")
    cos_angle = np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    return {
        "angle_deg": round(angle_deg, 4),
        "angle_rad": round(angle_rad, 6),
        "vertex": v.tolist(),
        "arm_a": a.tolist(),
        "arm_b": b.tolist(),
    }


@app.post("/api/measure-bone-length")
def measure_bone_length(
    case_id: str,
    point_proximal: str = Form(...),  # JSON [x,y,z]
    point_distal: str = Form(...),    # JSON [x,y,z]
):
    """Measure projected bone length along the segment between two points.

    Also returns the straight-line distance and the axis direction.
    """
    try:
        prox = np.array(json.loads(point_proximal), dtype=float)
        dist = np.array(json.loads(point_distal), dtype=float)
    except Exception:
        raise HTTPException(400, "invalid points")
    if prox.shape != (3,) or dist.shape != (3,):
        raise HTTPException(400, "each point must be [x,y,z]")
    vec = dist - prox
    length = float(np.linalg.norm(vec))
    axis = (vec / length).tolist() if length > 1e-12 else [0.0, 0.0, 1.0]
    mid = ((prox + dist) / 2.0).tolist()
    return {
        "bone_length_mm": length,
        "axis": axis,
        "proximal": prox.tolist(),
        "distal": dist.tolist(),
        "midpoint": mid,
    }


# ─── Export per-fragment STL endpoint ─────────────────────────────────────────

@app.post("/api/export-fragment-stl")
def export_fragment_stl(
    case_id: str,
    fragment_names: str = Form(""),  # JSON array of names, or empty for all
):
    """Return a combined STL with all (transformed) fragments positioned in space."""
    ft = _load_fragment_transforms(case_id)
    out_dir = OUTPUTS / case_id

    names = []
    if fragment_names:
        try:
            names = json.loads(fragment_names)
        except Exception:
            names = []

    if not names:
        names = list(ft.get("fragments", {}).keys())

    parts = []
    for name in names:
        finfo = ft.get("fragments", {}).get(name, {})
        # Prefer transformed version if available
        tf = finfo.get("transformed_url", "")
        if tf:
            tname = tf.split("/")[-1]
            tp = out_dir / tname
            if tp.exists():
                parts.append(load_mesh(tp))
                continue
        src = out_dir / finfo["file"]
        if src.exists():
            parts.append(load_mesh(src))

    if not parts:
        raise HTTPException(404, "no fragments found to export")

    combined = trimesh.util.concatenate(parts)
    combined.remove_unreferenced_vertices()
    out_path = out_dir / "fragments_combined.stl"
    combined.export(out_path)
    return {
        "url": f"/api/cases/{case_id}/outputs/fragments_combined.stl",
        "fragments_included": names,
        "info": mesh_info(combined),
    }


# ──────────────────────────────────────────────────────────────────────
# MPR (Multi-Planar Reconstruction) slice viewer helpers & endpoints
# ──────────────────────────────────────────────────────────────────────

# In-memory cache: case_id -> {"volume": np.ndarray, "spacing": (z,y,x), "meta": dict}
_mpr_volume_cache: dict[str, dict] = {}


def _get_cached_volume(case_id: str) -> dict:
    """Load and cache the DICOM volume for a case as a .npy file."""
    if case_id in _mpr_volume_cache:
        return _mpr_volume_cache[case_id]
    case_dir = UPLOADS / case_id
    npy_path = case_dir / "volume_cache.npy"
    meta_path = case_dir / "dicom_meta.json"
    if npy_path.exists() and meta_path.exists:
        vol = np.load(str(npy_path), mmap_mode="r")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        spacing = tuple(meta.get("spacing_mm", [1.0, 1.0, 1.0]))
        entry = {"volume": vol, "spacing": spacing, "meta": meta}
        _mpr_volume_cache[case_id] = entry
        return entry
    # Fall back to re-reading DICOM
    dicom_dir = case_dir / "dicom"
    if not dicom_dir.exists():
        raise HTTPException(404, "case has no DICOM data")
    vol, spacing, meta = load_dicom_series(dicom_dir)
    # Cache to disk for fast subsequent access
    np.save(str(npy_path), vol)
    entry = {"volume": vol, "spacing": spacing, "meta": meta}
    _mpr_volume_cache[case_id] = entry
    return entry


def _window_level_to_uint8(slice_2d: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """Apply HU window/level mapping to a 2D slice, return uint8 grayscale."""
    if window_width <= 0:
        window_width = 1.0
    lower = window_center - window_width / 2.0
    upper = window_center + window_width / 2.0
    result = np.clip((slice_2d - lower) / (upper - lower) * 255.0, 0, 255).astype(np.uint8)
    return result


def _get_mask_for_case(case_id: str) -> Optional[np.ndarray]:
    """Try to reconstruct the segmentation mask from the case volume + meta."""
    case_dir = UPLOADS / case_id
    mask_npy = case_dir / "mask_cache.npy"
    if mask_npy.exists():
        return np.load(str(mask_npy), mmap_mode="r")
    return None


def _reconstruct_mask(case_id: str, vol: np.ndarray, meta: dict) -> np.ndarray:
    """Reconstruct bone mask from volume using stored segmentation parameters."""
    hu_min = float(meta.get("threshold_hu", 250))
    hu_max = meta.get("threshold_max_hu", None)
    if hu_max is not None and hu_max > hu_min:
        mask = (vol >= hu_min) & (vol <= hu_max)
    else:
        mask = vol >= hu_min
    mask = mask.astype(np.float32)
    # Apply morphology if stored
    close_r = int(meta.get("morphology_closing_radius", 0))
    open_r = int(meta.get("morphology_opening_radius", 0))
    dil_r = int(meta.get("dilation_radius", 0))
    ero_r = int(meta.get("erosion_radius", 0))
    if close_r > 0 or open_r > 0 or dil_r > 0 or ero_r > 0:
        mask = _apply_morphology(mask, closing_radius=close_r, opening_radius=open_r,
                                  dilation_radius=dil_r, erosion_radius=ero_r)
    if meta.get("fill_holes", False):
        mask = _apply_morphology(mask, fill_holes=True)
    return (mask > 0.5).astype(np.uint8)


def _overlay_mask_on_grayscale(gray: np.ndarray, mask_slice: np.ndarray,
                                color: tuple[int, int, int] = (255, 60, 60),
                                alpha: float = 0.35) -> np.ndarray:
    """Overlay a binary mask on a grayscale image with given color and alpha."""
    rgba = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    rgba[..., 3] = 255
    m = mask_slice > 0
    if m.any():
        for c in range(3):
            rgba[m, c] = np.clip(
                (1 - alpha) * gray[m] + alpha * color[c], 0, 255
            ).astype(np.uint8)
        rgba[m, 3] = 255
    return rgba


@app.get("/api/cases/{case_id}/mpr-volinfo")
def mpr_volinfo(case_id: str):
    """Return volume dimensions and spacing for MPR calculations."""
    try:
        entry = _get_cached_volume(case_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to load volume: {e}")
    vol = entry["volume"]
    spacing = entry["spacing"]
    meta = entry["meta"]
    return {
        "dimensions": [int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])],  # slices, rows, cols
        "spacing_mm": list(spacing),  # z, y, x
        "shape": {
            "axial_slices": int(vol.shape[0]),
            "coronal_slices": int(vol.shape[1]),
            "sagittal_slices": int(vol.shape[2]),
        },
        "hu_range": [float(np.min(vol)), float(np.max(vol))],
        "series_uid": meta.get("series_uid", ""),
        "series_description": meta.get("series_description", ""),
    }


@app.get("/api/cases/{case_id}/slice-range")
def slice_range(case_id: str):
    """Return min/max slice indices for each axis."""
    try:
        entry = _get_cached_volume(case_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to load volume: {e}")
    vol = entry["volume"]
    return {
        "axial": {"min": 0, "max": int(vol.shape[0]) - 1, "count": int(vol.shape[0])},
        "coronal": {"min": 0, "max": int(vol.shape[1]) - 1, "count": int(vol.shape[1])},
        "sagittal": {"min": 0, "max": int(vol.shape[2]) - 1, "count": int(vol.shape[2])},
    }


@app.get("/api/cases/{case_id}/slice")
def get_slice(
    case_id: str,
    axis: str = "axial",
    index: int = 0,
    window_center: float = 40.0,
    window_width: float = 400.0,
    overlay_mask: bool = False,
):
    """Extract a single slice as PNG image with optional window/level and mask overlay.

    Parameters:
        axis: axial | coronal | sagittal
        index: slice index along the given axis
        window_center: HU center of the window
        window_width: HU width of the window
        overlay_mask: whether to overlay the segmentation mask
    """
    try:
        entry = _get_cached_volume(case_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to load volume: {e}")

    vol = entry["volume"]
    axis = axis.lower()

    # Extract the 2D slice from the 3D volume
    # Volume shape: (slices_z, rows_y, cols_x)
    if axis == "axial":
        max_idx = vol.shape[0] - 1
        index = max(0, min(index, max_idx))
        slice_2d = vol[index, :, :].astype(np.float32)
    elif axis == "coronal":
        max_idx = vol.shape[1] - 1
        index = max(0, min(index, max_idx))
        slice_2d = vol[:, index, :].astype(np.float32)
    elif axis == "sagittal":
        max_idx = vol.shape[2] - 1
        index = max(0, min(index, max_idx))
        slice_2d = vol[:, :, index].astype(np.float32)
    else:
        raise HTTPException(400, f"Invalid axis '{axis}'. Use axial, coronal, or sagittal.")

    # Apply window/level
    gray = _window_level_to_uint8(slice_2d, window_center, window_width)

    if overlay_mask:
        # Try to get or reconstruct mask
        mask = _get_mask_for_case(case_id)
        if mask is None:
            # Reconstruct on the fly (slower)
            try:
                mask = _reconstruct_mask(case_id, vol, entry["meta"])
            except Exception:
                mask = None
        if mask is not None:
            if axis == "axial":
                mask_slice = mask[index, :, :]
            elif axis == "coronal":
                mask_slice = mask[:, index, :]
            else:
                mask_slice = mask[:, :, index]
            img_rgba = _overlay_mask_on_grayscale(gray, mask_slice)
            img = Image.fromarray(img_rgba, mode="RGBA")
        else:
            img = Image.fromarray(gray, mode="L")
    else:
        img = Image.fromarray(gray, mode="L")

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")


# ══════════════════════════════════════════════════════════════════════
# Fase 4 — Mesh Boolean operations (Cork-style via trimesh)
# ══════════════════════════════════════════════════════════════════════

def _mesh_boolean(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    operation: str = "difference",
) -> trimesh.Trimesh:
    """Perform mesh boolean operation with fallback.

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
