import os, tempfile, zipfile
from pathlib import Path
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid
from app import dicom_to_bone_mesh

base=Path('/tmp/3dmedicalplanner_test_dicom')
if base.exists():
    import shutil; shutil.rmtree(base)
base.mkdir()
rows=64; cols=64; slices=48
cy,cx=rows/2,cols/2
for z in range(slices):
    y,x=np.ogrid[:rows,:cols]
    r=14 + 3*np.sin(z/7)
    mask=(x-cx)**2+(y-cy)**2 < r*r
    arr=np.full((rows,cols), -1000, dtype=np.int16)
    arr[mask]=700
    meta=FileMetaDataset()
    meta.MediaStorageSOPClassUID=CTImageStorage
    meta.MediaStorageSOPInstanceUID=generate_uid()
    meta.TransferSyntaxUID=ExplicitVRLittleEndian
    ds=FileDataset(str(base/f'slice_{z:03d}.dcm'), {}, file_meta=meta, preamble=b'\0'*128)
    ds.SOPClassUID=CTImageStorage; ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
    ds.Modality='CT'; ds.PatientName='Demo^Bone'; ds.PatientID='DEMO'
    ds.Rows=rows; ds.Columns=cols; ds.InstanceNumber=z+1
    ds.ImagePositionPatient=[0,0,float(z)]
    ds.PixelSpacing=[0.7,0.7]; ds.SliceThickness=1.0
    ds.SamplesPerPixel=1; ds.PhotometricInterpretation='MONOCHROME2'
    ds.BitsAllocated=16; ds.BitsStored=16; ds.HighBit=15; ds.PixelRepresentation=1
    ds.RescaleIntercept=0; ds.RescaleSlope=1
    ds.PixelData=arr.tobytes()
    ds.save_as(base/f'slice_{z:03d}.dcm')
mesh, meta = dicom_to_bone_mesh(base, threshold_hu=250, step_size=1,
                                morph_closing_radius=2, fill_holes_3d=True,
                                remove_small_islands_mm3=100, auto_crop=True)
print(len(mesh.vertices), len(mesh.faces), mesh.is_watertight, meta['slices'], meta['spacing_mm'])
print("v2:", meta.get('segmentation_v2', {}).get('morph_closing_radius'))
assert len(mesh.vertices) > 0 and len(mesh.faces) > 0
