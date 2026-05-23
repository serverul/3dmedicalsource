import json
import math
import os
import shutil
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pydicom
import trimesh
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
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


def load_dicom_series(dicom_dir: Path, selected_series_uid: str | None = None) -> tuple[np.ndarray, tuple[float, float, float], dict]:
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


# ── Segmentation Engine v2 ──────────────────────────────────────────────────

SEGMENTATION_PRESETS = {
    "long_bone": {
        "label": "Os lung (femur, tibia, humerus)",
        "hu_min": 250, "hu_max": 1800,
        "morph_closing": 3, "morph_opening": 1,
        "fill_holes_2d": True, "fill_holes_3d": True,
        "remove_small_islands_mm3": 200,
        "auto_crop": True, "align_axis": "long",
        "mesh_hole_fill": True, "mesh_repair": True,
        "smoothing": 3,
    },
    "skull": {
        "label": "Craniu / maxilofacial",
        "hu_min": 300, "hu_max": 3000,
        "morph_closing": 2, "morph_opening": 1,
        "fill_holes_2d": True, "fill_holes_3d": False,
        "remove_small_islands_mm3": 100,
        "auto_crop": True, "align_axis": "",
        "mesh_hole_fill": True, "mesh_repair": True,
        "smoothing": 2,
    },
    "spine": {
        "label": "Coloană / vertebre",
        "hu_min": 200, "hu_max": 1600,
        "morph_closing": 2, "morph_opening": 1,
        "fill_holes_2d": True, "fill_holes_3d": False,
        "remove_small_islands_mm3": 50,
        "auto_crop": True, "align_axis": "long",
        "mesh_hole_fill": True, "mesh_repair": True,
        "smoothing": 2,
    },
    "cortical": {
        "label": "Os cortical (HIGH density)",
        "hu_min": 600, "hu_max": 2000,
        "morph_closing": 1, "morph_opening": 1,
        "fill_holes_2d": False, "fill_holes_3d": False,
        "remove_small_islands_mm3": 50,
        "auto_crop": True, "align_axis": "",
        "mesh_hole_fill": False, "mesh_repair": False,
        "smoothing": 1,
    },
    "metal": {
        "label": "Metal artifact (implant, screw)",
        "hu_min": 2000, "hu_max": 10000,
        "morph_closing": 4, "morph_opening": 2,
        "fill_holes_2d": True, "fill_holes_3d": True,
        "remove_small_islands_mm3": 50,
        "auto_crop": True, "align_axis": "",
        "mesh_hole_fill": True, "mesh_repair": True,
        "smoothing": 2,
    },
}


def ball_3d(radius: int) -> np.ndarray:
    """Generate a 3D spherical structuring element of given radius."""
    if radius <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    d = 2 * radius + 1
    center = np.array([radius, radius, radius])
    grid = np.indices((d, d, d)) - center.reshape(3, 1, 1, 1)
    dist = np.sqrt(grid[0]**2 + grid[1]**2 + grid[2]**2)
    return dist <= radius


def apply_morphology(mask: np.ndarray, closing_radius: int = 0, opening_radius: int = 0) -> np.ndarray:
    """Apply binary closing and/or opening to a 3D boolean mask."""
    if closing_radius > 0:
        struct = ball_3d(closing_radius)
        mask = ndi.binary_closing(mask, structure=struct)
    if opening_radius > 0:
        struct = ball_3d(opening_radius)
        mask = ndi.binary_opening(mask, structure=struct)
    return mask


def fill_holes_2d_slices(mask: np.ndarray) -> np.ndarray:
    """Fill holes in each 2D slice independently (z-axis)."""
    out = mask.copy()
    for z in range(mask.shape[0]):
        out[z] = ndi.binary_fill_holes(out[z])
    return out


def fill_holes_3d_volume(mask: np.ndarray) -> np.ndarray:
    """Fill holes in 3D (background cavities fully enclosed by foreground)."""
    return ndi.binary_fill_holes(mask)


def remove_small_islands(mask: np.ndarray, min_voxels: int = 0, min_volume_mm3: float = 0, voxel_volume_mm3: float = 1.0) -> np.ndarray:
    """Remove connected components smaller than a threshold."""
    if min_voxels <= 0 and min_volume_mm3 <= 0:
        return mask
    labels, num = ndi.label(mask)
    if num <= 1:
        return mask
    counts = np.bincount(labels.ravel())
    threshold_voxels = max(min_voxels, int(min_volume_mm3 / max(voxel_volume_mm3, 0.001)))
    keep_labels = {i for i, c in enumerate(counts) if i > 0 and c >= threshold_voxels}
    if not keep_labels:
        # Keep the largest by default
        counts[0] = 0
        keep_labels = {int(counts.argmax())}
    return np.isin(labels, list(keep_labels))


def crop_to_bbox(mask: np.ndarray, margin_voxels: int = 5) -> tuple[np.ndarray, tuple[slice, ...]]:
    """Crop volume to bounding box of foreground voxels + margin."""
    coords = np.array(np.nonzero(mask))
    if coords.size == 0:
        return mask, (slice(None),) * mask.ndim
    mins = coords.min(axis=1) - margin_voxels
    maxs = coords.max(axis=1) + margin_voxels + 1
    slices = tuple(slice(max(0, int(m)), int(M)) for m, M in zip(mins, maxs))
    return mask[tuple(slices)], slices


def align_to_axis(mask: np.ndarray, target_axis: str = "long", spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
                  ) -> tuple[np.ndarray, dict]:
    """Compute PCA alignment of bone voxels. Returns rotation info for mesh stage.
    
    We don't rotate the volume (too expensive/interpolated). Instead we return
    PCA basis vectors so the mesh can be transformed after marching cubes.
    """
    coords = np.array(np.nonzero(mask)).T.astype(np.float64)  # Nx3 in voxel space
    if len(coords) < 10:
        return mask, {"aligned": False, "reason": "too few voxels"}
    # Scale to physical space
    coords[:, 0] *= spacing[0]  # z
    coords[:, 1] *= spacing[1]  # y
    coords[:, 2] *= spacing[2]  # x
    
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = centered.T @ centered / len(coords)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending by eigenvalue (longest axis first)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    # Ensure right-handed coordinate system
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1
    
    # Map target_axis to index
    target_idx = {"z": 0, "y": 1, "x": 2, "long": 0}.get(target_axis, 0)
    # We want the primary PCA axis to align to the target physical axis.
    # target_vec is e.g. [0,0,1] for x (CAD x = dicom col axis)
    if target_axis == "long":
        # Align primary PCA axis to CAD x-axis (highest variance)
        target_vec = np.array([0.0, 0.0, 1.0])
    elif target_axis == "x":
        target_vec = np.array([0.0, 0.0, 1.0])
    elif target_axis == "y":
        target_vec = np.array([0.0, 1.0, 0.0])
    elif target_axis == "z":
        target_vec = np.array([1.0, 0.0, 0.0])
    else:
        return mask, {"aligned": False, "reason": f"unknown target axis {target_axis}"}
    
    primary = eigvecs[:, 0]
    # Compute rotation that aligns primary to target_vec (rotation between two vectors)
    v = np.cross(primary, target_vec)
    s = np.linalg.norm(v)
    c = np.dot(primary, target_vec)
    if s < 1e-6:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    
    # Build 4x4 transform: rotate centroid-centered mesh, then translate back
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = centroid - R @ centroid
    
    return mask, {
        "aligned": True,
        "target_axis": target_axis,
        "pca_primary": primary.tolist(),
        "pca_centroid": centroid.tolist(),
        "rotation_matrix": R.tolist(),
        "transform_4x4": transform.tolist(),
    }


def repair_mesh(mesh: trimesh.Trimesh, fill_holes: bool = False, fix_normals: bool = False,
                decimation_factor: float = 0.0) -> trimesh.Trimesh:
    """Apply mesh-level cleanup operations."""
    if fix_normals and not mesh.is_watertight:
        try:
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass
    if fill_holes:
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass
    if decimation_factor > 0 and decimation_factor < 1:
        try:
            target = max(12, int(len(mesh.faces) * (1 - decimation_factor)))
            mesh = mesh.simplify_quadric_decimation(target)
        except Exception:
            pass
    return mesh


def resolve_preset(preset_name: str, overrides: dict) -> dict:
    """Merge preset defaults with user overrides."""
    preset = SEGMENTATION_PRESETS.get(preset_name, {})
    params = {
        "threshold_hu": 250.0,
        "threshold_max_hu": None,
        "morph_closing_radius": 0,
        "morph_opening_radius": 0,
        "fill_holes_2d": False,
        "fill_holes_3d": False,
        "remove_small_islands_mm3": 0,
        "auto_crop": True,
        "align_axis": "",
        "mesh_hole_fill": False,
        "mesh_repair_normals": False,
        "smoothing_iterations": 2,
        "decimation_factor": 0.0,
    }
    # Apply preset
    if preset:
        params["threshold_hu"] = preset.get("hu_min", params["threshold_hu"])
        params["threshold_max_hu"] = preset.get("hu_max")
        params["morph_closing_radius"] = preset.get("morph_closing", 0)
        params["morph_opening_radius"] = preset.get("morph_opening", 0)
        params["fill_holes_2d"] = preset.get("fill_holes_2d", False)
        params["fill_holes_3d"] = preset.get("fill_holes_3d", False)
        params["remove_small_islands_mm3"] = preset.get("remove_small_islands_mm3", 0)
        params["auto_crop"] = preset.get("auto_crop", True)
        params["align_axis"] = preset.get("align_axis", "")
        params["mesh_hole_fill"] = preset.get("mesh_hole_fill", False)
        params["mesh_repair_normals"] = preset.get("mesh_repair", False)
        params["smoothing_iterations"] = preset.get("smoothing", 2)
    # User overrides win
    params.update(overrides)
    return params


def dicom_to_bone_mesh(
    dicom_dir: Path,
    threshold_hu: float = 250.0,
    step_size: int = 1,
    series_uid: str | None = None,
    threshold_max_hu: float | None = None,
    keep_largest: bool = True,
    smoothing_iterations: int = 2,
    # v2 segmentation params
    preset: str = "",
    morph_closing_radius: int = 0,
    morph_opening_radius: int = 0,
    fill_holes_2d: bool = False,
    fill_holes_3d: bool = False,
    remove_small_islands_mm3: float = 0.0,
    auto_crop: bool = True,
    align_axis: str = "",
    mesh_hole_fill: bool = False,
    mesh_repair_normals: bool = False,
    decimation_factor: float = 0.0,
) -> tuple[trimesh.Trimesh, dict]:
    vol, spacing, meta = load_dicom_series(dicom_dir, selected_series_uid=series_uid)
    hu_min_actual = float(np.min(vol)); hu_max_actual = float(np.max(vol))
    
    # Resolve preset if given
    if preset:
        preset_params = resolve_preset(preset, {
            "threshold_hu": threshold_hu,
            "threshold_max_hu": threshold_max_hu,
            "smoothing_iterations": smoothing_iterations,
        })
        threshold_hu = preset_params["threshold_hu"]
        threshold_max_hu = preset_params["threshold_max_hu"]
        morph_closing_radius = preset_params["morph_closing_radius"]
        morph_opening_radius = preset_params["morph_opening_radius"]
        fill_holes_2d = preset_params["fill_holes_2d"]
        fill_holes_3d = preset_params["fill_holes_3d"]
        remove_small_islands_mm3 = preset_params["remove_small_islands_mm3"]
        auto_crop = preset_params["auto_crop"]
        align_axis = preset_params["align_axis"]
        mesh_hole_fill = preset_params["mesh_hole_fill"]
        mesh_repair_normals = preset_params["mesh_repair_normals"]
        smoothing_iterations = preset_params["smoothing_iterations"]
        decimation_factor = preset_params["decimation_factor"]
    
    if threshold_hu <= hu_min_actual or threshold_hu >= hu_max_actual:
        raise ValueError(f"threshold {threshold_hu} outside volume HU range {meta['hu_min']}..{meta['hu_max']}")
    
    # ── Step 1: HU threshold → binary mask ──
    if threshold_max_hu is not None and threshold_max_hu > threshold_hu:
        mask = (vol >= threshold_hu) & (vol <= threshold_max_hu)
    else:
        mask = vol >= threshold_hu
    
    # ── Step 2: Morphology ──
    if morph_closing_radius > 0 or morph_opening_radius > 0:
        mask = apply_morphology(mask, closing_radius=morph_closing_radius, opening_radius=morph_opening_radius)
    
    # ── Step 3: Fill holes ──
    if fill_holes_2d:
        mask = fill_holes_2d_slices(mask)
    if fill_holes_3d:
        mask = fill_holes_3d_volume(mask)
    
    # ── Step 4: Remove small islands ──
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    if remove_small_islands_mm3 > 0 or keep_largest:
        min_voxels = max(int(remove_small_islands_mm3 / max(voxel_volume, 0.001)), 0)
        if keep_largest:
            labels, num = ndi.label(mask)
            if num > 1:
                counts = np.bincount(labels.ravel())
                counts[0] = 0
                largest = int(counts.argmax())
                keep = np.isin(labels, [largest])
                if min_voxels > 0:
                    # Also keep others above threshold
                    for i, c in enumerate(counts):
                        if i > 0 and i != largest and c >= min_voxels:
                            keep |= np.isin(labels, [i])
                mask = keep
            elif min_voxels > 0:
                mask = remove_small_islands(mask, min_voxels=min_voxels, voxel_volume_mm3=voxel_volume)
        else:
            mask = remove_small_islands(mask, min_voxels=min_voxels, voxel_volume_mm3=voxel_volume)
    
    # ── Step 5: ROI crop ──
    crop_slices = None
    if auto_crop:
        mask, crop_slices = crop_to_bbox(mask, margin_voxels=5)
    
    # ── Step 6: Axis alignment (compute PCA transform for mesh stage) ──
    alignment_info = {"aligned": False}
    if align_axis:
        _, alignment_info = align_to_axis(mask, target_axis=align_axis, spacing=spacing)
    
    # ── Step 7: Marching cubes ──
    src_float = mask.astype(np.float32)
    verts, faces, normals, values = measure.marching_cubes(
        src_float,
        level=0.5,
        spacing=spacing,
        step_size=max(1, int(step_size)),
        allow_degenerate=False,
    )
    # skimage returns z,y,x coordinates. Convert to x,y,z for CAD-like display.
    verts_xyz = verts[:, [2, 1, 0]]
    mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    
    # ── Step 8: Apply PCA alignment to mesh ──
    if alignment_info.get("aligned"):
        try:
            tf = np.array(alignment_info["transform_4x4"])
            mesh.apply_transform(tf)
        except Exception:
            pass
    
    # ── Step 9: Mesh cleanup ──
    mesh = repair_mesh(mesh, fill_holes=mesh_hole_fill, fix_normals=mesh_repair_normals,
                       decimation_factor=decimation_factor)
    
    # ── Step 10: Smoothing ──
    if smoothing_iterations > 0:
        try:
            trimesh.smoothing.filter_laplacian(mesh, lamb=0.35, iterations=int(smoothing_iterations))
        except Exception:
            pass
    
    # ── Build meta ──
    meta.update({
        "threshold_hu": threshold_hu,
        "threshold_max_hu": threshold_max_hu,
        "keep_largest": bool(keep_largest),
        "smoothing_iterations": int(smoothing_iterations),
        "step_size": int(step_size),
        "segmentation_v2": {
            "preset": preset or None,
            "morph_closing_radius": morph_closing_radius,
            "morph_opening_radius": morph_opening_radius,
            "fill_holes_2d": fill_holes_2d,
            "fill_holes_3d": fill_holes_3d,
            "remove_small_islands_mm3": remove_small_islands_mm3,
            "auto_crop": auto_crop,
            "align_axis": align_axis or None,
            "mesh_hole_fill": mesh_hole_fill,
            "mesh_repair_normals": mesh_repair_normals,
            "decimation_factor": decimation_factor,
            "alignment": alignment_info,
            "crop_slices": [str(s) for s in crop_slices] if crop_slices else None,
        },
        "generated_vertices": int(len(mesh.vertices)),
        "generated_faces": int(len(mesh.faces)),
        "warning": "CT→bone mesh via Segmentation Engine v2 (HU threshold + morphology + cleanup). Not medical-device validated.",
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
    threshold_max_hu: float = Form(0.0),
    preset: str = Form(""),
    morph_closing_radius: int = Form(0),
    morph_opening_radius: int = Form(0),
    fill_holes_2d: bool = Form(False),
    fill_holes_3d: bool = Form(False),
    remove_small_islands_mm3: float = Form(0.0),
    auto_crop: bool = Form(True),
    align_axis: str = Form(""),
    mesh_hole_fill: bool = Form(False),
    mesh_repair_normals: bool = Form(False),
    smoothing_iterations: int = Form(2),
    decimation_factor: float = Form(0.0),
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
            dicom_dir, threshold_hu=threshold_hu, step_size=step_size,
            threshold_max_hu=threshold_max_hu if threshold_max_hu > 0 else None,
            preset=preset,
            morph_closing_radius=morph_closing_radius,
            morph_opening_radius=morph_opening_radius,
            fill_holes_2d=fill_holes_2d,
            fill_holes_3d=fill_holes_3d,
            remove_small_islands_mm3=remove_small_islands_mm3,
            auto_crop=auto_crop,
            align_axis=align_axis,
            mesh_hole_fill=mesh_hole_fill,
            mesh_repair_normals=mesh_repair_normals,
            smoothing_iterations=smoothing_iterations,
            decimation_factor=decimation_factor,
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
    threshold_max_hu: float = Form(0.0),
    preset: str = Form(""),
    morph_closing_radius: int = Form(0),
    morph_opening_radius: int = Form(0),
    fill_holes_2d: bool = Form(False),
    fill_holes_3d: bool = Form(False),
    remove_small_islands_mm3: float = Form(0.0),
    auto_crop: bool = Form(True),
    align_axis: str = Form(""),
    mesh_hole_fill: bool = Form(False),
    mesh_repair_normals: bool = Form(False),
    smoothing_iterations: int = Form(2),
    decimation_factor: float = Form(0.0),
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
            dicom_dir, threshold_hu=threshold_hu, step_size=step_size,
            threshold_max_hu=threshold_max_hu if threshold_max_hu > 0 else None,
            preset=preset,
            morph_closing_radius=morph_closing_radius,
            morph_opening_radius=morph_opening_radius,
            fill_holes_2d=fill_holes_2d,
            fill_holes_3d=fill_holes_3d,
            remove_small_islands_mm3=remove_small_islands_mm3,
            auto_crop=auto_crop,
            align_axis=align_axis,
            mesh_hole_fill=mesh_hole_fill,
            mesh_repair_normals=mesh_repair_normals,
            smoothing_iterations=smoothing_iterations,
            decimation_factor=decimation_factor,
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
    preset: str = Form(""),
    morph_closing_radius: int = Form(0),
    morph_opening_radius: int = Form(0),
    fill_holes_2d: bool = Form(False),
    fill_holes_3d: bool = Form(False),
    remove_small_islands_mm3: float = Form(0.0),
    auto_crop: bool = Form(True),
    align_axis: str = Form(""),
    mesh_hole_fill: bool = Form(False),
    mesh_repair_normals: bool = Form(False),
    decimation_factor: float = Form(0.0),
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
            preset=preset,
            morph_closing_radius=morph_closing_radius,
            morph_opening_radius=morph_opening_radius,
            fill_holes_2d=fill_holes_2d,
            fill_holes_3d=fill_holes_3d,
            remove_small_islands_mm3=remove_small_islands_mm3,
            auto_crop=auto_crop,
            align_axis=align_axis,
            mesh_hole_fill=mesh_hole_fill,
            mesh_repair_normals=mesh_repair_normals,
            decimation_factor=decimation_factor,
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
