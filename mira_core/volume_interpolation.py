from abc import ABC, abstractmethod

import SimpleITK

from mira_core.volume_info import *


def is_integer_type(value):
    return isinstance(value, (int, np.integer))


class ImageInterpolation(ABC):

    _class_registry = {}
    _logger = logging.getLogger(__name__)

    def __init__(self, description: str, **kwargs):
        super().__init__(**kwargs)
        self._name = self.__class__.__name__
        self._description = description



    def __init_subclass__(cls, **kwargs):
        super.__init_subclass__(**kwargs)

        #register child classes
        cls._class_registry[cls.__name__] = cls
        cls._logger.debug(f'Registered: {cls.__name__}')

    @classmethod
    def create_from_config(cls, config:dict):
        class_type = config.get('class')
        subclass = cls._class_registry.get(class_type)
        if not subclass:
            raise ValueError(f"Unknown class type: {class_type}")
        args = config.get('args',{})
        return subclass(**args)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    @abstractmethod
    def valid_sitk_types(self):
        # if returns none all types are valid
        # otherwise return a list of supported types
        pass

    @property
    @abstractmethod
    def valid_np_types(self):
        pass

    def check_sitk_type(self, im: sitk.Image) -> sitk.Image:
        if isinstance(im, sitk.Image):
            if self.valid_sitk_types is not None:
                if im.GetPixelID() not in self.valid_sitk_types:
                    raise TypeError(f"Invalid sitk type: {im}, {im.GetPixelID()}")
        else:
            raise TypeError(f"Invalid type: {type(im)}")
        return im

    def check_np_type(self, im) -> np.ndarray:
        if isinstance(im, np.ndarray):
            if self.valid_np_types is not None:
                if im.dtype not in self.valid_np_types:
                    raise TypeError(f"Invalid numpy type: {im}, {im.dtype}")
        else:
            raise TypeError(f"Invalid type: {type(im)}")
        return im


class ImageTypes(Enum):
    NUMPY = 0
    SITK = 1


class VolumeResize(ImageInterpolation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def resize(self, factor: Union[np.ndarray, list], image: sitk.Image) -> sitk.Image:
        pass

    def resize_numpy(
        self, factor: Union[np.ndarray, list], image_array: np.ndarray, image_info: VolumeInformation = None, image_array_order=VolumeIndexType.XYZ
    ) -> (np.ndarray, VolumeInformation):
        original_type = image_array.dtype
        image_array = self.check_np_type(image_array)

        if image_info is None:
            image_info = VolumeInformation.from_array(image_array)

        im_sitk = image_info.get_sitk_image(image_array, image_array_order)

        up_im = self.resize(factor, im_sitk)
        up_im_info = image_info.from_sitk_image(up_im)

        array = sitk.GetArrayFromImage(up_im).astype(original_type)
        array = np.moveaxis(array, [0, 1, 2], SITK_ARRAY_INDICES.move_to(image_array_order))
        return array, up_im_info

    def resize_1d(self, factor: Union[int, float], image: sitk.Image, slice_normal: VolumeAxis = VolumeAxis.Z) -> sitk.Image:
        ind = SITK_IMAGE_INDICES.get_index(slice_normal)
        factor_3d = np.ones(3)
        factor_3d[ind] = factor
        return self.resize(factor_3d, image)

    def resize_numpy_1d(
        self,
        factor: Union[np.integer, int],
        image_array: np.ndarray,
        image_info: VolumeInformation = None,
        slice_normal: VolumeAxis = VolumeAxis.Z,
        image_array_order: VolumeIndexType = SITK_ARRAY_TYPE,
    ) -> (sitk.Image, VolumeInformation):
        """



        :param factor: resample factor, must result in an image that is has integer size
        :param mask: an image representing a contour either bool, or integer type (uint8, uint16, or int16)
        :param slice_normal: the normal to the slice the contour
        :param image_array_order: the ordering of the image array e.g. default image_array[z,y,x]
        :return: returns an image upsampled along the normal direction of the type input, in the same array format
        """

        # BLAAAH Duplication of code....
        image_array = self.check_np_type(image_array)

        if image_info is None:
            image_info = VolumeInformation.from_array(image_array, image_array_order)
        im_sitk = image_info.get_sitk_image(image_array, image_array_order)
        up_im = self.resize_1d(factor, im_sitk, slice_normal)
        up_im_info = image_info.from_sitk_image(up_im)
        array = sitk.GetArrayFromImage(up_im).astype(image_array.dtype)
        # move the volume back to original order
        array = np.moveaxis(array, [0, 1, 2], SITK_ARRAY_INDICES.move_to(image_array_order))
        return array, up_im_info


class VolumeResample(ImageInterpolation):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def resample_from_image_info(self, image: sitk.Image, new_image_info: VolumeInformation, transform: sitk.Transform = sitk.Transform()):
        image_2 = new_image_info.get_empty_sitk_image()
        return self.resample(image, image_2, transform)

    @abstractmethod
    def resample(self, image: sitk.Image, resample_2_image: sitk.Image, transform: sitk.Transform = sitk.Transform()):
        pass

    # this is default behaviour if your funciton doesn't use sitk images but operates on numpy data be smart
    # and overload this to save on memory
    def resample_numpy(self, image_array: np.ndarray, image_info: VolumeInformation, resample_image_info: VolumeInformation) -> (np.ndarray, VolumeInformation):
        input_image_sitk = image_info.get_sitk_image(image_array)
        resample_image_sitk = resample_image_info.get_empty_sitk_image()
        image_itk = self.resample(input_image_sitk, resample_image_sitk)
        return sitk.GetArrayFromImage(image_itk), VolumeInformation.from_sitk_image(image_itk)

class DefaultPixelValueType(Enum):
    ZERO = 0
    CUSTOM = (1,)
    MIN_IMAGE_VALUE = (2,)
    MAX_IMAGE_VALUE = (3,)

#todo not so happy with multiple inheritence makes a bit of uncertaintity around _logger and class registry

class sitkVolumeInterpolation(VolumeResize, VolumeResample):
    default_sitk_interpolators = {"Linear": sitk.sitkLinear, "NearestNeighbor": sitk.sitkNearestNeighbor, "Gaussian": sitk.sitkGaussian, "BSpline": sitk.sitkBSpline}


    def __init__(self, sitk_interpolator: int):
        super().__init__(description="A General purpose interpolator using SimpleITK resample")
        self._logger.debug(f"Creating sitkVolumeInterpolation with {sitk_interpolator}")
        self._sitk_interpolators = list(self.default_sitk_interpolators.values())
        self._sitk_interpolator = self.set_sitk_interpolator(sitk_interpolator)
        self._default_pixel_value = DefaultPixelValueType.MIN_IMAGE_VALUE
        self._custom_pixel_value = 0

    @property
    def valid_sitk_types(self):
        # support all types
        return None

    @property
    def valid_np_types(self):
        # support all Types
        return None

    @property
    def sitk_interpolator(self):
        return self._sitk_interpolator

    def set_sitk_interpolator(self, sitk_interpolator):
        if sitk_interpolator not in self._sitk_interpolators:
            raise ValueError(f"Invalid sitk interpolator: {sitk_interpolator}")
        self._sitk_interpolator = sitk_interpolator
        return self._sitk_interpolator

    def set_default_pixel_value(self, default_pixel_value: DefaultPixelValueType, custom=None):
        self._default_pixel_value = default_pixel_value
        if self._default_pixel_value == DefaultPixelValueType.CUSTOM:
            self._custom_pixel_value = custom

    def resample(self, image: sitk.Image, resample_2_image: sitk.Image, transform: sitk.Transform = sitk.Transform()):
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetInterpolator(self._sitk_interpolator)
        resample_filter.SetOutputPixelType(sitk.sitkUnknown)
        resample_filter.SetReferenceImage(resample_2_image)
        resample_filter.SetTransform(transform)
        resample_filter.SetDefaultPixelValue(self.get_default_pixel_value(image))
        resampled_image = resample_filter.Execute(image)

        return resampled_image

    def resize(self, factor: Union[np.ndarray, list], image: sitk.Image) -> sitk.Image:
        assert isinstance(image, sitk.Image)

        self.check_sitk_type(image)

        original_origin = np.array(image.GetOrigin())
        original_spacing = np.array(image.GetSpacing())
        original_size = np.array(image.GetSize())
        original_direction = image.GetDirection()

        out_size = original_size * factor
        if np.sum(np.abs(out_size - np.round(out_size))):
            self._logger.warning(f"Scaling factor results in an output image that is not a integer size, scaled size {out_size}")

        out_spacing = original_spacing / factor
        out_size = tuple(int(i) for i in out_size)  # typing is messed with this has to be int not np.int

        # Use image's internal method to get true physical corner
        dim = len(original_size)
        original_corner = image.TransformContinuousIndexToPhysicalPoint([-0.5] * np.ones(dim))

        # New origin keeps the corner fixed and shifts to the center of new [0,0,0] voxel
        orientation_matrix = np.array(original_direction).reshape((dim, dim))
        output_origin = original_corner + orientation_matrix @ (0.5 * out_spacing)

        print(f"Original Corner {original_corner}, ")

        # old way ignores direction
        # original_corner = original_origin - (original_spacing / 2.0)
        # out_origin = original_corner + np.array(out_spacing) / 2

        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(self._sitk_interpolator)
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(original_direction)
        resample.SetOutputOrigin(output_origin)
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(self.get_default_pixel_value(image))

        return resample.Execute(image)

    def get_default_pixel_value(self, image: sitk.Image):
        if self._default_pixel_value == DefaultPixelValueType.CUSTOM:
            return self._custom_pixel_value
        elif self._default_pixel_value == DefaultPixelValueType.MIN_IMAGE_VALUE:
            return float(np.min(sitk.GetArrayFromImage(image)))
        elif self._default_pixel_value == DefaultPixelValueType.MAX_IMAGE_VALUE:
            return float(np.max(SimpleITK.GetArrayFromImage(image)))
        else:
            return 0


class sitkNearestNeighborInterp(sitkVolumeInterpolation):
    def __init__(self):
        super().__init__(sitk.sitkNearestNeighbor)

class sitkBSplineInterp(sitkVolumeInterpolation):
    def __init__(self):
        super().__init__(sitk.sitkBSpline)

class sitkLinearInterp(sitkVolumeInterpolation):
    def __init__(self):
        super().__init__(sitk.sitkLinear)


class sitkGaussianInterp(sitkVolumeInterpolation):
    def __init__(self):
        super().__init__(sitk.sitkGaussian)
