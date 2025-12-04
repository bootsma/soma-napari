import logging
from enum import Enum
from typing import Union

import itk
import numpy as np
import SimpleITK as sitk

from dicom_utils import get_ref_image_series_uid, get_series_image_info, load_dicom_image, load_dicom_images


def convert_itk_to_sitk(itk_image):
    """
    Converts an ITK image to a SimpleITK image.

    Args:
        itk_image (itk.itkImagePython.itkImageSS3): The input ITK image.

    Returns:
        sitk.Image: The converted SimpleITK image.
    """
    # Convert the ITK image to a NumPy array
    np_array = itk.GetArrayFromImage(itk_image)

    # Create a SimpleITK image from the NumPy array
    sitk_image = sitk.GetImageFromArray(np_array)

    # Copy the origin, spacing, and direction from the ITK image to the SimpleITK image
    sitk_image.SetOrigin(np.array(itk_image.GetOrigin()))
    sitk_image.SetSpacing(np.array(itk_image.GetSpacing()))
    sitk_image.SetDirection(np.array(itk_image.GetDirection()).flatten())

    return sitk_image


def convert_sitk_to_itk(sitk_image):
    """
    Converts a SimpleITK image to an ITK image.

    Args:
        sitk_image (sitk.Image): The input SimpleITK image.

    Returns:
        itk.itkImagePython.itkImageSS3: The converted ITK image.
    """
    # Convert the SimpleITK image to a NumPy array
    np_array = sitk.GetArrayFromImage(sitk_image)

    # Create an ITK image from the NumPy array
    itk_image = itk.GetImageFromArray(np_array)

    # Copy the origin, spacing, and direction from the SimpleITK image to the ITK image
    itk_image.SetOrigin(sitk_image.GetOrigin())
    itk_image.SetSpacing(sitk_image.GetSpacing())
    itk_image.SetDirection(np.array(sitk_image.GetDirection()).reshape(3, 3))

    return itk_image


class Unit(Enum):
    m = "m"
    cm = "cm"
    mm = "mm"


class VolumeAxis(Enum):
    X = "x"
    Y = "y"
    Z = "z"


class VolumeIndexType(Enum):
    XYZ = "xyz"
    XZY = "xzy"
    ZXY = "zxy"
    ZYX = "zyx"
    YXZ = "yxz"
    YZX = "yzx"


SITK_ARRAY_TYPE = VolumeIndexType.ZYX
SITK_IMAGE_TYPE = VolumeIndexType.XYZ


class VolumeIndices:
    def __init__(self, volume_type: VolumeIndexType):
        self.volume_type = volume_type

    def get_index(self, axis: Union[str, VolumeAxis]):
        if isinstance(axis, VolumeAxis):
            axis = axis.value
        if isinstance(axis, str) and len(axis) == 1 and axis.lower() in ["x", "y", "z"]:
            return self.volume_type.value.find(axis.lower())
        else:
            raise IndexError("Key not valid")

    @property
    def type(self):
        return self.volume_type

    def __getitem__(self, key):
        return self.get_index(key)

    def to(self, volume_type: VolumeIndexType):
        return [self.get_index(axis) for axis in volume_type.value]

    def move_to(self, volume_type: VolumeIndexType):
        """
        used for np.moveaxis maps the indices in order to new order
        e.g. np.moveaxis(array, [0,1,2], self.move_to(VolumeIndexType.YZX))
             xyz->zxy   [1,2,0]
        :param volume_type:
        :return:
        """
        if isinstance(volume_type, VolumeIndices):
            volume_type = volume_type.type
        new_indices = VolumeIndices(volume_type)
        return [new_indices.get_index(axis) for axis in self.volume_type.value]

    def move_axis(self, array: np.ndarray, dest_index_type=SITK_ARRAY_TYPE):
        np.moveaxis(array, [0, 1, 2], self.move_to(dest_index_type))

    @property
    def x(self):
        return self.volume_type.value.find("x")

    @property
    def y(self):
        return self.volume_type.value.find("y")

    @property
    def z(self):
        return self.volume_type.value.find("z")


SITK_ARRAY_INDICES = VolumeIndices(SITK_ARRAY_TYPE)
SITK_IMAGE_INDICES = VolumeIndices(SITK_IMAGE_TYPE)


class UnitConverter:
    conversion_factors = {(Unit.m, Unit.cm): 100, (Unit.m, Unit.mm): 1000, (Unit.cm, Unit.m): 0.01, (Unit.cm, Unit.mm): 10, (Unit.mm, Unit.m): 0.001, (Unit.mm, Unit.cm): 0.1}

    @classmethod
    def convert(cls, value, from_unit, to_unit):
        if from_unit == to_unit:
            return value

        key = (from_unit, to_unit)
        if key in cls.conversion_factors:
            return value * cls.conversion_factors[key]
        else:
            raise ValueError(f"Conversion from {from_unit} to {to_unit} is not supported.")

    @classmethod
    def get_conversion_factor(cls, from_unit, to_unit):
        return UnitConverter.convert(1, from_unit, to_unit)


class VolumeInformation:
    """
     Information for a 3D Volume

    pixel 0,0,0  in a spacing of (1,1,1) and origin of (0,0,0) spans intervals  -0.5<=x<0.5, -0.5<=y<0.5,-0.5<=z<0.5
    """

    def __init__(
        self,
        origin: Union[list, np.ndarray, tuple],
        spacing: Union[list, np.ndarray, tuple],
        orientation: Union[list, np.ndarray, tuple],
        size: Union[list, np.ndarray, tuple],
        units=Unit.mm,
        **kwargs,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._origin = np.array(origin)
        self._spacing = np.array(spacing)
        self._orientation = np.array(orientation).reshape(3, 3)
        self.is_orientation_matrix_is_unit()
        assert self.is_orthonormal_direction()
        self._inv_orientation = np.linalg.inv(self._orientation)
        self._size = np.array(size)
        self._units = units
        # precomple scaling into matrix
        self._physical_to_index = self._inv_orientation.T / self._spacing
        self._index_to_physical = (self._orientation @ np.diag(self._spacing)).T

        # ImageInfo is always stored as XYZ(e.g. origin, spacing, orientation, size all x,y,z
        # and operations (e.g. resize, crop) assume input as xyz
        # it is currently an underlying assumption it could be changed to using  self.__image_info_indices
        # to account for variations
        self.__vol_info_indices = VolumeIndices(VolumeIndexType.XYZ)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __eq__(self, other):
        """
        Compares two VolumeInformation objects based on key spatial attributes.
        """
        if not isinstance(other, VolumeInformation):
            return False

        # Compare origin, spacing, orientation, and size using numpy.array_equal
        origin_equal = np.allclose(self._origin, other._origin)
        spacing_equal = np.allclose(self._spacing, other._spacing)
        orientation_equal = np.allclose(self._orientation, other._orientation)
        size_equal = np.array_equal(self._size, other._size)

        # Compare units
        units_equal = self._units == other._units

        # Return True only if all specified attributes are equal
        return origin_equal and spacing_equal and orientation_equal and size_equal and units_equal

    def is_orientation_matrix_is_unit(self, tol=1e-6):
        norm = np.linalg.norm(self._orientation, axis=0)
        if not np.allclose(norm, 1.0, atol=tol):
            self._logger.warning("orientation matrix is not unitary, this may cause unexpected behavior for i,j,k. Please check your data. Norm:{norm}")
            return False
        else:
            return True

    def is_orthonormal_direction(self, tol=1e-6):
        dim = 3
        direction = self._orientation

        # Check pairwise orthogonality
        for i in range(dim):
            for j in range(i + 1, dim):
                dot = np.dot(direction[:, i], direction[:, j])
                if not np.isclose(dot, 0.0, atol=tol):
                    return False

        return True

    def is_axis_aligned(self, tol=1e-6):
        direction = self._orientation
        # Each row or column should be one-hot with ±1 and all others ~0
        for i in range(3):
            axis = direction[:, i]
            axis = axis / np.linalg.norm(axis)
            is_one_hot = np.isclose(np.abs(axis), 1.0, atol=tol) | np.isclose(axis, 0.0, atol=tol)
            if not np.all(is_one_hot):
                return False
            if not np.isclose(np.linalg.norm(axis), 1.0, atol=tol):
                return False

        return True

    def copy(self):
        return VolumeInformation(self.origin, self.spacing, self.orientation, self.size, self.units)

    @classmethod
    def from_sitk_image(cls, image: sitk.Image, units=Unit.mm) -> "VolumeInformation":
        return cls(origin=image.GetOrigin(), spacing=image.GetSpacing(), orientation=image.GetDirection(), size=image.GetSize(), units=units)

    @classmethod
    def from_itk_image(cls, image, units=Unit.mm) -> "VolumeInformation":
        np_array = itk.array_from_image(image)
        # note here that xyz shape is flip of np_array shape b/c underlying data is zyx
        return cls(origin=image.GetOrigin(), spacing=image.GetSpacing(), orientation=image.GetDirection(), size=np.flip(np_array.shape), units=units)

    @classmethod
    def from_rtstruct_ref_image(cls, rtstruct_file_path: str, dicom_path: str) -> "VolumeInformation":
        """

        :param rtstruct_file_path: the path to the rtstruct data
        :param dicom_path: the path that contains the image series data associated with the rtstruct data
        :return: the image informations associated with the data in the series the contours were drawn on
        """
        ref_series_uid = get_ref_image_series_uid(rtstruct_file_path)
        return cls.from_dicom_image(dicom_path=dicom_path, series_uid=ref_series_uid)

    @classmethod
    def from_dicom_image(cls, dicom_path: str, series_uid: str = None, slice_thickness_check: bool = False, single_series_check: bool = False) -> "VolumeInformation":
        """

        :param dicom_path: path to series
        :param series_uid: if None it assumes only a single series UID is in directory
        :param slice_thickness_check: if true it will make sure all slices in series have same thickness
        :param single_series_check: if true this will make sure the directory only has a single series
        in it, only valid if series_uid is set to None. otherwise it assumes first image in series is
         representative of entire series
        :return:
        """

        return cls(**(get_series_image_info(dicom_path, series_uid, slice_thickness_check, single_series_check)))

    @classmethod
    def from_array(cls, image_array: np.ndarray, units=Unit.mm, array_index_type: VolumeIndexType = SITK_ARRAY_TYPE) -> "VolumeInformation":
        """
        Generate image info from an array assume SITK ordering ZYX
        :param image_array:
        :param units:
        :param array_index_type:
        :return:
        """
        assert image_array.ndim == 3
        if array_index_type != SITK_ARRAY_TYPE:
            indices = VolumeIndices(array_index_type)
            image_array = np.moveaxis(image_array, [0, 1, 2], indices.move_to(SITK_ARRAY_TYPE))
        if image_array.dtype == bool:
            image_array = image_array.astype(np.int8)
        image = sitk.GetImageFromArray(image_array)
        return cls.from_sitk_image(image, units)

    @classmethod
    def get_volume_info_and_array(self, rtstruct_path:str, dicom_image_set_path:str) -> ("VolumeInformation", np.ndarray):
        """
        Load an RT Struct and DICOM Image Set and return the Volume Information and Array
        :param rtstruct_path: Path to RT Struct file
        :param dicom_image_set_path:l path to DICOM Image Set directory
        :return: VolumeInformation, numpy array of image data
        """
        ref_series_uid = get_ref_image_series_uid(rtstruct_path)

        sitk_image = load_dicom_images(dicom_image_set_path, series_instance_uid=ref_series_uid)
        return VolumeInformation.from_sitk_image(sitk_image), sitk.GetArrayFromImage(sitk_image)

    def get_empty_sitk_image(self, data_type=np.float32) -> sitk.Image:
        image = sitk.GetImageFromArray(np.zeros(np.flip(self._size), data_type))
        image.SetSpacing(self._spacing.tolist())
        image.SetOrigin(self._origin.tolist())
        image.SetDirection(self._orientation.ravel().astype(float))
        return image

    def get_num_voxels(self):
        return self._size.prod()

    def get_sitk_image(self, array: np.ndarray, array_index_type: VolumeIndexType = SITK_ARRAY_TYPE) -> sitk.Image:
        """Converts a numpy array to a sitk image with the same metadata as this ImageInformation object.
        Assumes the array is in zyx format...
        """
        if isinstance(array_index_type, VolumeIndices):
            array_index_type = array_index_type.type

        assert isinstance(array_index_type, VolumeIndexType)
        if array_index_type != SITK_ARRAY_TYPE:
            indices = VolumeIndices(array_index_type)
            array = np.moveaxis(array, [0, 1, 2], indices.move_to(SITK_ARRAY_TYPE))

        if array.dtype == bool:
            self._logger.debug("Converting bool array to int for sitk image purposes")
            array = array.astype(np.uint8)

        image = sitk.GetImageFromArray(array)

        if not np.array_equal(np.array(image.GetSize()), self._size):
            raise ValueError("array size does not match image size")
        image.SetSpacing(self._spacing)
        image.SetOrigin(self._origin)
        image.SetDirection(self._orientation.ravel().astype(float))
        return image

    def get_itk_image(self, array: np.ndarray, array_index_type: VolumeIndexType = SITK_ARRAY_TYPE):
        return convert_sitk_to_itk(self.get_sitk_image(array, array_index_type))

    """ use get_resize_image_info ... this function needs to offset teh origin
    def resize(self, scaling_factor: Union[int, list, np.ndarray]):


        self._spacing = self._spacing / scaling_factor
        self._size = self._size * scaling_factor
    """

    def get_image_extent(self):
        return self.top_corner, self.bottom_corner

    def crop_array_to_physical_extent(
        self,
        min_extent: Union[tuple, list, np.array],
        max_extent: Union[tuple, list, np.array],
        array: np.ndarray,
        volume_index: VolumeIndexType,
        pad_pixels: Union[tuple, list, np.array] = None,
    ) -> (np.ndarray, "VolumeInformation"):
        crop_image_info = self.copy()
        min_pix, max_pix = crop_image_info.crop_info_to_physical_extent(min_extent, max_extent, pad_pixels)
        index_swap = self.__vol_info_indices.to(volume_index)
        min_pix = min_pix[index_swap]
        max_pix = max_pix[index_swap]

        crop_array = array[
            min_pix[0] : max_pix[0] + 1,
            min_pix[1] : max_pix[1] + 1,
            min_pix[2] : max_pix[2] + 1,
        ]
        return crop_array, crop_image_info

    def pad_info_with_pixels(self, xyz_pixel_pad):
        """
        Uniform padding of image by Nx,Ny,Nz =xyz_pixel_pad on all sides
        :param xyz_pad:
        :return:
        """
        pixels = np.array(xyz_pixel_pad).astype(int)
        self._size += pixels * 2
        self._origin = self.continuous_index_to_physical_point(-1 * pixels)

    def crop_info_to_physical_extent(self, min_extent: Union[tuple, list, np.array], max_extent: Union[tuple, list, np.array], pad_pixels: Union[tuple, list, np.array] = None):
        """

        :param min_extent:  min x,y,z of volume
        :param max_extent:  max x,y,z of volume
        :param pad_pixels:  or None, recommend use (1,1,1)
        :return: returns min and max index into volume as x,y,z, which can be used like
           volume_xyz[min_extent[0]:max_extent[0],
                      min_extent[1]:max_extent[1],
                      min_extent[2]:max_extent[2]]
        """

        if not self.is_axis_aligned():
            self._logger.warning("Axes must be aligned with XYZ in some form (e.g. ZYX, YXZ) for crop to make sense with physical extent")

        if pad_pixels is None:
            pad_pixels = np.array([0, 0, 0])
        pixel_min = self.physical_point_to_index(min_extent)
        pixel_max = self.physical_point_to_index(max_extent)

        # the direction of the image might be flipped on some axis
        pixel_min_extent = np.minimum(pixel_min, pixel_max) - pad_pixels
        pixel_max_extent = np.maximum(pixel_max, pixel_min) + pad_pixels

        return self.crop_info_to_pixel_extent(pixel_min_extent, pixel_max_extent)

    def crop_info_to_pixel_extent(
        self,
        min_extent: Union[tuple, list, np.array],
        max_extent: Union[tuple, list, np.array],
    ):
        # keep it inside the image
        pixel_min_extent = np.maximum(min_extent, np.zeros(min_extent.shape))
        pixel_max_extent = np.minimum(max_extent, self._size - 1)

        if not np.array_equal(pixel_min_extent, min_extent) or not np.array_equal(pixel_max_extent, max_extent):
            self._logger.warning(
                f"crop_info_to_pixel_extent(): input extent was outside image\nNEW: min:{pixel_min_extent}, max:{pixel_max_extent}\nINPUT: min:{min_extent}, max:{max_extent}"
            )

        if np.any(pixel_min_extent == pixel_max_extent):
            self._logger.warning("crop_info_to_pixel_extent(): input extent results in 0 in one dimension")

        # adjust size and origin properly
        self._size = pixel_max_extent - pixel_min_extent + 1
        self._origin = self.continuous_index_to_physical_point(pixel_min_extent)

        return pixel_min_extent.astype(int), pixel_max_extent.astype(int)

    def get_resized_image_info(self, scaling: Union[list, np.ndarray, tuple]) -> "VolumeInformation":
        """
            Returns a new ImageInformation object with the same metadata as this one, but scaled by the given factor.
        :param scaling: A list of three floats representing the scaling factors for each dimension.
        :return: A new ImageInformation object with the updated metadata.
        """
        if scaling is None:
            return self.copy()
        if not isinstance(scaling, np.ndarray):
            scaling = np.array(scaling)
        corner = self.top_corner
        size = (self._size * scaling).astype(int)
        spacing = self._spacing / scaling
        origin = corner + self._orientation @ (0.5 * spacing)
        return VolumeInformation(size=size, spacing=spacing, origin=origin, orientation=self._orientation.ravel(), units=self._units)

    def get_empty_numpy_array(self, volume_index: VolumeIndexType, data_type=np.float32) -> np.ndarray:
        """
            Returns an empty numpy array with the same dimensions and data type as this ImageInformation object.
        :param volume_index: remember SimpleITK format is ZYX
        :param data_type:
        :return:
        """

        indices = self.__vol_info_indices.to(volume_index)
        return np.zeros(self._size[indices], dtype=data_type)

    def physical_point_to_continuous_index(self, point: Union[list, np.ndarray]) -> np.ndarray:
        """
        For the simple case where origin is (0,0) and spacing is (1,1) the pixel [0,0]
        spans the x={-0.5, 0.5-epsilon}, y={0.5, 0.5-epsilon}
        :param point:
        :return:
        """
        # single point : (self._inv_orientation@(point-self._origin))/self._spacing =R^(-1)@(p-o)
        # if orientation is orthonormal: ((point - self._origin).dot(self._orientation) / self._spacing)
        # (point-self._origin)@self._inv_orientation.T/self._spacing
        return (point - self._origin) @ self._physical_to_index

    def physical_point_to_index(self, point: Union[list, np.ndarray]):
        return np.floor(self.physical_point_to_continuous_index(point) + 0.5).astype(int)

    def continuous_index_to_physical_point(self, index: Union[list, np.ndarray]):
        # Transform is R*diag(s)*x+x_o, x_o=origin, R=orientation matrix, s=scale
        # single point : self._orientation@(index*self._spacing)+self.origin = R*diag(scale)*index+o
        # if orientation is orthonormal:(index*self._spacing).dot(self._inv_orientation)+self._origin
        return index @ self._index_to_physical + self.origin

    def index_to_physical_point(self, index: Union[list, np.ndarray]):
        return self.continuous_index_to_physical_point(index)

    def set_units(self, units: Unit):
        if units == self._units:
            return
        else:
            f = UnitConverter.get_conversion_factor(self._units, units)
            self._origin = self._origin * f
            self._spacing = self._spacing * f
            self._units = units

    @property
    def i(self):
        """
        :return: x-axis unit vector
        """
        return self._orientation[:, 0]

    def j(self):
        """
        :return: y-axis unit vector
        """
        return self._orientation[:, 1]

    def k(self):
        """
        :return: z-axis unit vector
        """
        return self._orientation[:, 2]

    @property
    def voxel_volume(self) -> float:
        return np.prod(self._spacing)

    @property
    def image_information_index_type(self) -> VolumeIndexType:
        return self.__vol_info_indices.type

    @property
    def origin(self):
        return self._origin

    @property
    def top_corner(self):
        # self.origin-self._spacing/2 only works if no direction involved
        return self.continuous_index_to_physical_point([-0.5, -0.5, -0.5])

    @property
    def bottom_corner(self):
        return self.continuous_index_to_physical_point(self._size - 0.5)

    @property
    def spacing(self):
        return self._spacing

    @property
    def direction(self):
        return self._orientation.ravel()

    @property
    def orientation(self):
        return self._orientation

    @property
    def size(self):
        return self._size

    @property
    def nx(self):
        return self._size[0]

    @property
    def ny(self):
        return self._size[1]

    @property
    def nz(self):
        return self._size[2]

    @property
    def units(self):
        return self._units

    def get_config(self):
        return {
            "size": self._size.tolist(),
            "spacing": self._spacing.tolist(),
            "origin": self._origin.tolist(),
            "orientation": self._orientation.ravel().tolist(),
            "units": str(self.units),
        }

    def __str__(self):
        return f"ImageInformation(origin={self._origin}, spacing={self._spacing}, direction={self._orientation.ravel()}, size={self._size})"

    @classmethod
    def from_dicom(cls, exam_dicom):
        pass


if __name__ == "__main__":
    vol_array = np.zeros([30, 20, 10])
    izyx = VolumeInformation.from_array(vol_array)  # assume ZYX
    ixyz = VolumeInformation.from_array(vol_array, VolumeIndexType.XYZ)
    ixyz = VolumeInformation.from_array(vol_array, VolumeIndexType.XYZ)

    sitk_im = sitk.GetImageFromArray(vol_array)
    itk_im = itk.image_from_array(vol_array)
    itk_im_chk = convert_sitk_to_itk(sitk_im)

    xyz = VolumeIndices(VolumeIndexType.XYZ)
    print(xyz.to(VolumeIndexType.YXZ))
    print(xyz.to(VolumeIndexType.XYZ))
    print(xyz.to(VolumeIndexType.ZYX))

    yxz = VolumeIndices(VolumeIndexType.YXZ)
    original = np.array([0, 1, 2])

    indices = yxz.to(VolumeIndexType.ZYX)
    print(indices)
    rindices = yxz.move_to(VolumeIndexType.YXZ)
    print(rindices)

    sz = np.array([100, 20, 10])  # xyz
    ind = xyz.to(VolumeIndexType.ZYX)
    print(ind, sz[ind])

    info = VolumeInformation.from_array(np.zeros(sz))
    print(info)

#   x = np.zeros(5,4,3)
#  xnew1 = np.moveaxis(x,[0,1,2],[1,2,0])

# print(SITK_ARRAY_INDICES.to(VolumeIndexType.YXZ))
