# Research Notes

These notes capture the external references discussed during the initial build and how they affect implementation.

## Bonabyte / BonaPlanner observation

Bonabyte's public site is mostly marketing/lead-generation. The relevant product idea is BonaPlanner-like 3D planning:

```txt
CT/DICOM → 3D model → planning → guide/implant design → manufacturing output
```

We are not cloning the website. The useful product is a surgical CAD workflow.

## Existing tools / product positioning

### Horos

Free Mac DICOM viewer. Covers a lot of 2D viewing, measurements, and basic 3D rendering.

### VPOP Pro

Covers veterinary 2D orthopedic planning/templates.

### 3D Slicer

Open-source medical imaging platform. Good candidate for a future extension or companion workflow:

- DICOM import
- segmentation
- VTK/ITK ecosystem
- markups/measurements
- STL export

### Product gap

Value is not another 2D viewer. Value is:

- 3D CT-based segmentation
- virtual osteotomy
- fragment repositioning
- conformal cutting/drill guides
- patient-specific plates/implants

## Frontiers Veterinary Science 2022 article

Reference: `https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2022.971318/full`

Relevant workflow:

```txt
CT / CBCT
→ DICOM
→ segmentation by HU/volume growing/manual refinement
→ 3D mesh STL
→ virtual surgical planning
→ osteotomy/resection planning
→ surgical guide / drill guide / custom implant design
→ 3D print
→ surgery
```

Important implications:

- HU range selection matters; bad threshold makes anatomy artificially bigger/smaller.
- Beam-hardening and artifacts must be removed/refined.
- Guides must fit a precise contour of patient anatomy.
- Instrument dimensions matter: saw blade, bur, piezotome, drill/pin tolerances.
- Drill guides and screw trajectories are part of real clinical workflow.
- Patient-specific metal implants/plates require exact screw dimensions and manufacturing constraints.

Implementation requirements derived:

- segmentation controls and validation
- conformal guide contact patch
- instrument library
- drill/screw trajectory editor
- material/manufacturing metadata
- case iteration and surgeon/designer collaboration

## PMC4499051 — cortical bone segmentation

Reference: `PMC4499051`, DOI `10.1118/1.4923753`.

Title: **Automated cortical bone segmentation for multirow-detector CT imaging with validation and application to human studies**.

Core point: simple thresholding is not enough for accurate cortical bone segmentation.

Problems identified:

- limited resolution
- low signal-to-noise ratio
- cortical pores
- trabecular/cortical transition
- partial volume effects
- complex boundary between marrow/endosteal/periosteal regions

Algorithm structure in paper:

1. bone filling, alignment, ROI computation
2. cortical bone segmentation:
   - detect marrow space and possible pores
   - compute cortical thickness
   - detect recession points
   - confirm/fill true pores
   - detect endosteal boundary
   - delineate cortical bone

Reported validation:

- 95.1% agreement vs manual cortical bone outlining on MD-CT
- 88.5% vs registered high-resolution micro-CT
- repeat-scan ICC 0.98

Implementation implications:

- Add bone fill / hole fill.
- Add morphology and connected component cleanup.
- Add ROI crop.
- Add axis alignment/PCA.
- Add cortical shell / dense cortical segmentation mode.
- Eventually add thickness map for screw/plate planning.

## Suggested algorithm direction

For MVP v2:

```txt
HU mask
→ connected components
→ keep selected/largest component
→ binary closing/opening
→ fill holes
→ remove small islands
→ crop ROI
→ marching cubes
→ smooth/decimate
→ validate watertight/manifold/extents
```

For later:

```txt
periosteal surface + endosteal/marrow boundary
→ cortical shell segmentation
→ thickness map
→ screw purchase analysis
```
