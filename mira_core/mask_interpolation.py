import itk

from mira_core.volume_interpolation import *


class sitkMaskIntepolation(sitkVolumeInterpolation):
    valid_sitk_types_list = [sitk.sitkUInt8, sitk.sitkUInt16, sitk.sitkInt16]
    valid_numpy_types_list = [bool, np.uint8, np.uint16, np.int16]

    # these values were optimized on 2d images, may want to optimize on 3d shapes
    threshold_defaults = {
        sitk.sitkLinear: 0.4,
        sitk.sitkBSpline: 0.3,
        sitk.sitkGaussian: 0.5,
        sitk.sitkNearestNeighbor: None,  # doesn't matter
    }

    def __init__(self, sitk_interpolator, threshold_overide=None, intensity_scale_overide=None):
        self.threshold = 0.5
        self.intensity_scale = 100
        super().__init__(sitk_interpolator)
        if threshold_overide is not None:
            self.threshold = threshold_overide
        if intensity_scale_overide is not None:
            self.intensity_scale = intensity_scale_overide

    @property
    def valid_sitk_types(self):
        # support all types
        return self.valid_sitk_types_list

    @property
    def valid_np_types(self):
        # support all Types
        return self.valid_numpy_types_list

    def set_sitk_interpolator(self, sitk_interpolator):
        self.threshold = self.threshold_defaults.get(sitk_interpolator, -1)
        if self.threshold == -1:
            self._logger.error("Invalid interpolator for mask interpolation")
            raise Exception("Invalid interpolator for mask interpolation")

        return super().set_sitk_interpolator(sitk_interpolator)

    def set_threshold(self, value):
        self._logger.debug("Overriding default threshold, setting threshold to {}".format(value))
        self.threshold = value

    def set_intensity_scale(self, value):
        self._logger.debug(f"Setting intensity scale to {value}")
        self.intensity_scale = value

    def threshold_mask(self, mask_image: sitk.Image) -> sitk.Image:
        threshold = self.threshold
        insideValue = 1
        outsideValue = 0
        image_array = sitk.GetArrayFromImage(mask_image)
        min_value = np.min(image_array)
        max_value = np.max(image_array)

        mask_image = (mask_image - min_value) / (max_value - min_value)
        lower_threshold = threshold
        # lower_threshold = threshold * (max_value - min_value) + min_value
        return sitk.BinaryThreshold(mask_image, lowerThreshold=lower_threshold, upperThreshold=float(max_value + 1), insideValue=insideValue, outsideValue=outsideValue)

    def resize(self, factor: Union[np.ndarray, list], image: sitk.Image) -> sitk.Image:
        if self.threshold is not None:
            image = image * self.intensity_scale

        image = super().resize(factor, image)

        if self.threshold is not None:
            image = self.threshold_mask(image)

        return image


class itkMorphologicalMaskInterpolator(VolumeResize):
    """
    This is best used if you are increasing your slice thickness in one dimension, it was designed to fill in
    gaps in skipped contours.
    Can only make the volume/mask bigger not smaller

    """

    valid_sitk_types_list = [sitk.sitkUInt8, sitk.sitkUInt16, sitk.sitkInt16]
    valid_numpy_types_list = [bool, np.uint8, np.uint16, np.int16]

    def __init__(self):
        super().__init__(description="A interpolator using itk morphological contour interpolator")

    @property
    def valid_sitk_types(self):
        # support all types
        return self.valid_sitk_types_list

    @property
    def valid_np_types(self):
        # support all Types
        return self.valid_numpy_types_list

    def resize(self, factor: Union[np.ndarray, list], image: sitk.Image) -> sitk.Image:
        return convert_itk_to_sitk(self.__morphological_interp(factor, convert_sitk_to_itk(image)))

    def __morphological_interp(self, factor: Union[np.ndarray, list], image, flip_interp_order=False):
        factor = np.array(factor)

        array = itk.array_from_image(image)
        array_info = VolumeInformation.from_itk_image(image)

        if not isinstance(array, np.ndarray) or len(array.shape) != 3:
            raise ValueError("Input must be a 3D numpy array.")

        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("Scaling factor must be an integer.")
        if np.sum(factor < 1) > 0:
            raise ValueError("Scaling factor must be an integer > 0")

        # assume sitk array ordering e.g. z,y,x
        # do each dimension separately because we are using morphological filter to fill inbetween
        curr_array = array
        curr_info = array_info
        curr_image = image

        flip_order = flip_interp_order
        if flip_order:
            factor_tmp = np.flip(factor)
        else:
            factor_tmp = factor
        for i, i_factor in enumerate(factor_tmp):
            if i_factor != 1:
                if i_factor > 1:
                    # sitk volume is z,y,x
                    if flip_order:
                        curr_sitk_index = i
                    else:
                        curr_sitk_index = 2 - i

                    new_sz = np.array(curr_array.shape)
                    new_sz[curr_sitk_index] = new_sz[curr_sitk_index] * i_factor

                    assert np.isclose(int(i_factor), i_factor)

                    slices = tuple(slice(None, None, int(i_factor)) if j == curr_sitk_index else slice(None, None, 1) for j in range(3))
                    new_array = np.zeros(new_sz, dtype=curr_array.dtype)
                    new_array[slices] = curr_array

                    prev_array = curr_array
                    curr_array = new_array

                    # here we have x,y,z
                    image_resize = np.ones(3)
                    image_resize[i] = i_factor
                    curr_info = curr_info.get_resized_image_info(image_resize)

                    # make an sitk image
                    curr_image = curr_info.get_itk_image(curr_array)
                    curr_image = itk.morphological_contour_interpolator(curr_image)
                    curr_array = itk.array_from_image(curr_image)
                    """
                    f, ax = plt.subplots(2, 3)
                    ax[0, 0].imshow(new_array[int(new_array.shape[0] / 2), :, :])
                    ax[0, 1].imshow(new_array[:, int(new_array.shape[1] / 2), :])
                    ax[0, 2].imshow(new_array[:, :, int(new_array.shape[2] / 2)])
                    ax[1, 0].imshow(curr_array[int(curr_array.shape[0] / 2), :, :])
                    ax[1, 1].imshow(curr_array[:, int(curr_array.shape[1] / 2), :])
                    ax[1, 2].imshow(curr_array[:, :, int(curr_array.shape[2] / 2)])
                    plt.show()
                    """

        return curr_image


"""

This is a simple example showing how morphological contour interpolator works
It is best used to interpolate if increasing slice thickness (e.g. 1-dimension is changing)

import numpy as np
import itk
import matplotlib.pyplot as plt

from mira.data.image_info import ImageInformation


def create_np_test_image():

    im_array = np.zeros([30,30,30],dtype=np.int16)
    pad = 4
    for i in range(pad,im_array.shape[0]-pad):
        if i%2:
            im_array[i,+i:-i,+i:-i] = 1
    itk_im = itk.image_from_array(im_array  )
    return itk_im
if __name__ == '__main__':
    itk_im = create_np_test_image()
    i_np = itk.array_from_image(itk_im)
    try:
        itk_interp = itk.morphological_contour_interpolator(itk_im)
        itk_interp_np = itk.array_from_image(itk_interp)
        f,ax = plt.subplots(2,1)
        ax[0].imshow(i_np[:,int(i_np.shape[1]/2),:])
        ax[1].imshow(itk_interp_np[:, int(i_np.shape[1] / 2), :])
        plt.show()
"""
