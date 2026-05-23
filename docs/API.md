# API

Base URL during MVP: `http://127.0.0.1:8121` or Tailscale IP.

## Health

### `GET /api/health`

Returns app name/version.

## Import

### `POST /api/upload`

Upload STL/OBJ/PLY mesh.

Form fields:

- `file`: `.stl`, `.obj`, `.ply`

Returns:

- `case_id`
- `mesh_url`
- `info`

### `POST /api/sample`

Generates/loads synthetic long-bone sample.

### `POST /api/upload-dicom`

Upload DICOM ZIP or `.dcm`.

Form fields:

- `file`
- `threshold_hu` default `250`
- `step_size` default `1`

Returns generated `input.stl` and DICOM metadata.

### `POST /api/upload-dicom-folder`

Upload full DICOM folder from browser `webkitdirectory`.

Form fields:

- `files`: repeated multipart files
- `threshold_hu` default `250`
- `step_size` default `1`

Returns generated `input.stl` and DICOM metadata.

## Case / DICOM

### `GET /api/recent-cases?limit=10`

Lists recent local runtime cases. Used for `Load latest CT processed` button.

### `GET /api/cases/{case_id}/dicom-info`

Reads DICOM folder and returns selected/largest series metadata plus `available_series`.

### `POST /api/cases/{case_id}/regenerate-mesh`

Regenerate bone mesh from already uploaded DICOM folder.

Form fields:

- `threshold_hu` default `250`
- `threshold_max_hu` default `3000`
- `step_size` default `1`
- `series_uid` optional
- `keep_largest` default `true`
- `smoothing_iterations` default `2`

Returns:

- `case_id`
- `mesh_url`
- `info`
- `dicom`

## Files

### `GET /api/cases/{case_id}/input.stl`

Serves input/generated STL.

### `GET /api/cases/{case_id}/outputs/{name}`

Serves generated output files from allowlisted suffixes.

### `GET /api/cases/{case_id}/outputs/plan.json`

Serves cut plan JSON.

## Osteotomy / fragment transform

### `POST /api/cut`

Form fields:

- `case_id`
- `axis`: `x`, `y`, `z`
- `offset_mm`
- `rotate_side`: `positive` or `negative`
- `rotate_axis`: `x`, `y`, `z`
- `rotate_deg`
- `translate_x`, `translate_y`, `translate_z`

Returns positive/negative fragment metadata and output URLs:

- `positive.stl`
- `negative.stl`
- `combined_repositioned.stl`
- `plan.json`

## Guides

### `POST /api/guide`

Generate parametric cutting guide.

Form fields:

- `case_id`
- `axis`
- `offset_mm`
- `length_mm`
- `height_mm`
- `rail_thickness_mm`
- `slot_width_mm`
- `pin_radius_mm`
- `pin_spacing_mm`
- `pin_sleeve_outer_mm`

Returns:

- `cutting_guide.stl`
- `guide.json`

### `POST /api/conformal-guide`

Generate first-pass conformal contact pad / guide.

Form fields:

- `case_id`
- `axis`
- `offset_mm`
- `contact_side`: `positive` or `negative`
- `length_mm`
- `height_mm`
- `thickness_mm`
- `clearance_mm`
- `resolution_u`
- `resolution_v`
- `include_slot_rails`

Returns:

- `conformal_contact_pad.stl`
- `conformal_cutting_guide.stl`
- `conformal_guide.json`
