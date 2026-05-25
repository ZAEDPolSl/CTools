import nibabel as nib
import numpy as np
import SimpleITK as sitk

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


def sitk_to_nifti(sitk_image: sitk.Image) -> nib.Nifti1Image:
    """
    Convert a SimpleITK image to an in-memory nibabel NIfTI image.

    The voxel array is transposed from SimpleITK's z,y,x order to nibabel's
    x,y,z order and the affine is built from the SimpleITK spacing, origin,
    and direction matrix while converting coordinates from LPS to RAS.
    """

    array_zyx = sitk.GetArrayFromImage(sitk_image)
    array_xyz = np.transpose(array_zyx, (2, 1, 0))

    direction = np.asarray(sitk_image.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(sitk_image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(sitk_image.GetOrigin(), dtype=np.float64)

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = LPS_TO_RAS @ direction @ np.diag(spacing)
    affine[:3, 3] = LPS_TO_RAS @ origin
    return nib.Nifti1Image(array_xyz, affine)


def nifti_to_sitk(nifti_image: nib.Nifti1Image) -> sitk.Image:
    """
    Convert an in-memory nibabel NIfTI image to a SimpleITK image.

    The voxel array is transposed from nibabel's x,y,z order to SimpleITK's
    z,y,x order and the affine is decomposed into spacing, origin, and
    direction while converting coordinates from RAS back to LPS.
    """

    array_xyz = np.asanyarray(nifti_image.dataobj)
    array_zyx = np.transpose(array_xyz, (2, 1, 0))
    sitk_image = sitk.GetImageFromArray(array_zyx)

    affine = np.asarray(nifti_image.affine, dtype=np.float64)
    direction_spacing = LPS_TO_RAS @ affine[:3, :3]
    spacing = np.linalg.norm(direction_spacing, axis=0)
    direction = np.divide(
        direction_spacing,
        spacing,
        out=np.eye(3, dtype=np.float64),
        where=spacing > 0,
    )
    origin = LPS_TO_RAS @ affine[:3, 3]

    sitk_image.SetSpacing(tuple(spacing))
    sitk_image.SetDirection(tuple(direction.reshape(-1)))
    sitk_image.SetOrigin(tuple(origin))
    return sitk_image
