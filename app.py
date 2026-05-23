import json
import math
import os
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
import trimesh
from scipy.spatial import cKDTree
from scipy.ndimage import binary_closing, binary_opening, binary_dilation, binary_erosion, label as ndi_label
from skimage import measure
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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
    "long_bone": {
        "threshold_hu": 250,
        "threshold_max_hu": 3000,
        "morphology_closing_radius": 2,
        "morphology_opening_radius": 0,
        "fill_holes": True,
        "min_island_volume_mm3": 500.0,
        "smoothing_iterations": 2,
        "decimation_target_faces": None,
    },
    "skull": {
        "threshold_hu": 200,
        "threshold_max_hu": 3000,
        "morphology_closing_radius": 3,
        "morphology_opening_radius": 1,
        "fill_holes": True,
        "min_island_volume_mm3": 200.0,
        "smoothing_iterations": 3,
        "decimation_target_faces": None,
    },
    "spine": {
        "threshold_hu": 220,
        "threshold_max_hu": 3000,
        "morphology_closing_radius": 2,
        "morphology_opening_radius": 1,
        "fill_holes": True,
        "min_island_volume_mm3": 100.0,
        "smoothing_iterations": 2,
        "decimation_target_faces": None,
    },
    "cortical": {
        "threshold_hu": 400,
        "threshold_max_hu": 3000,
        "morphology_closing_radius": 1,
        "morphology_opening_radius": 0,
        "fill_hu": True,
        "min_island_volume_mm3": 300.0,
        "smoothing_iterations": 1,
        "decimation_target_faces": None,
    },
    "metal": {
        "threshold_hu": 200,
        "threshold_max_hu": 8000,
        "morphology_closing_radius": 3,
        "morphology_opening_radius": 2,
        "fill_holes": True,
        "min_island_volume_mm3": 500.0,
        "smoothing_iterations": 2,
        "decimation_target_faces": None,
    },
}


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
                   repair: bool = True) -> trimesh.Trimesh:
    """Post-process mesh: smooth, decimate, repair normals/watertight."""
    out = mesh.copy()
    if smoothing_iterations > 0:
        try:
            trimesh.smoothing.filter_laplacian(out, lamb=0.35, iterations=int(smoothing_iterations))
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
    # --- Segmentation Engine v2 parameters ---
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
) -> tuple[trimesh.Trimesh, dict]:
    """Convert DICOM CT series to bone mesh with Segmentation Engine v2.

    New v2 pipeline:
      HU mask → ROI crop → morphology → remove small islands → keep largest
      → fill holes → marching cubes → smooth → decimate → repair → [PCA align]
    """
    # Apply preset defaults then override with explicit params
    if preset and preset.lower() in SEGMENTATION_PRESETS:
        preset_cfg = SEGMENTATION_PRESETS[preset.lower()]
        threshold_hu = preset_cfg["threshold_hu"]
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
    hu_min_actual = float(np.min(vol)); hu_max_actual = float(np.max(vol))
    if threshold_hu <= hu_min_actual or threshold_hu >= hu_max_actual:
        raise ValueError(f"threshold {threshold_hu} outside volume HU range {meta['hu_min']}..{meta['hu_max']}")
    if threshold_max_hu is not None and threshold_max_hu > threshold_hu:
        mask = (vol >= threshold_hu) & (vol <= threshold_max_hu)
        src = mask.astype(np.float32)
    else:
        src = None  # use raw HU for marching_cubes level

    # --- ROI crop (on raw HU volume before threshold) ---
    crop_offset = (0, 0, 0)
    if roi_crop and src is not None:
        src, crop_offset, orig_shape = _roi_crop(src, spacing)
        # Adjust spacing for cropped volume — marching_cubes uses cropped array
        # We keep spacing as-is because voxel size hasn't changed, just the array bounds

    # --- Morphology ---
    if src is not None:
        src = _apply_morphology(
            src,
            closing_radius=morphology_closing_radius,
            opening_radius=morphology_opening_radius,
            dilation_radius=dilation_radius,
            erosion_radius=erosion_radius,
            fill_holes=False,  # fill holes after island removal
        )

    # --- Remove small islands ---
    if src is not None and min_island_volume_mm3 > 0:
        src = _remove_small_islands(src, spacing, min_volume_mm3=min_island_volume_mm3)

    # --- Keep largest connected component ---
    if keep_largest and src is not None:
        labels = measure.label(src > 0.5, connectivity=1)
        if labels.max() > 0:
            counts = np.bincount(labels.ravel()); counts[0] = 0
            src = (labels == int(counts.argmax())).astype(np.float32)

    # --- Fill holes ---
    if fill_holes and src is not None:
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
        offset_phys = np.array([crop_offset[0] * spacing[0], crop_offset[1] * spacing[1], crop_offset[2] * spacing[2]])
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
        "pca_align": pca_align,
        "pca_transform": pca_transform.tolist() if pca_transform is not None else None,
        "decimation_target_faces": decimation_target_faces,
        "repair_mesh": repair_mesh,
        "active_preset": active_preset,
        "generated_vertices": int(len(mesh.vertices)),
        "generated_faces": int(len(mesh.faces)),
        "mesh_validation": manifold_report,
        "warning": "Segmentation Engine v2. Needs manual validation for surgical use.",
    })
    return mesh, meta


def make_sample_bone() -> Path:
    path = SAMPLES / "sample_long_bone.stl"
    if path.exists():
        return path
    # Stylized long bone: cylinder shaft + two ellipsoid ends, not anatomical, just test geometry.
    shaft = trimesh.creation.cylinder(radius=13, height=120, sections=64)
    shaft.apply_transform(trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0]))
    top = trimesh.creation.uv_sphere(segments=64, ring_count=32)
    top.apply_scale([20, 16, 18])
    top.apply_translation([0, 60, 0])
    bot = trimesh.creation.uv_sphere(segments=64, ring_count=32)
    bot.apply_scale([22, 15, 16])
    bot.apply_translation([0, -60, 0])
    mesh = trimesh.util.concatenate([shaft, top, bot])
    mesh.export(path)
    return path


@app.get("/")
def root():
    return FileResponse(STATIC / "index.html")


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
        )
    except Exception as e:
        shutil.rmtree(case_dir, ignore_errors=True)
        raise HTTPException(400, f"DICOM conversion failed: {e}")
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
        )
    except Exception as e:
        shutil.rmtree(case_dir, ignore_errors=True)
        raise HTTPException(400, f"DICOM folder conversion failed: {e}")
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
    case_id: str = Form(...),
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
    case_id: str = Form(...),
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
    case_id: str = Form(...),
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


app.mount("/static", StaticFiles(directory=STATIC), name="static")
