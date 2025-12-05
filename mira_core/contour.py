import logging
import time
from typing import List, Tuple, Union


#todo move includes into functions so dependencies are hidden if a user only wants to use say rasterio for mask gen
import cv2
import numpy as np
import pydicom
import rasterio
import SimpleITK as sitk

from numpy.typing import NDArray
from PIL import Image, ImageDraw
from rasterio.features import geometry_mask
from scipy.interpolate import splev, splprep
from skimage.draw import polygon
from matplotlib.path import Path
from shapely.geometry import Polygon

from mira_core.volume_info import SITK_ARRAY_INDICES, SITK_ARRAY_TYPE, SITK_IMAGE_INDICES, Unit, UnitConverter, VolumeAxis, VolumeIndexType, VolumeIndices, VolumeInformation
from mira_core.dicom_utils import get_rtstruct_contour_roi_names
from mira_core.mask_interpolation import itkMorphologicalMaskInterpolator
from mira_core.volume_interpolation import VolumeResize

# serial version of loader
def RtStructCombinedMaskLoader( ref_image_info: VolumeInformation,
                                rtstruct_dicom_data: Union[str, pydicom.dataset.FileDataset],
                                roi_index_map: dict,
                                VolumeOrder:VolumeIndexType=SITK_ARRAY_TYPE,
                                progress_callback=None) -> np.ndarray:

    logger = logging.getLogger(__name__)
    start = time.perf_counter()

    if isinstance(rtstruct_dicom_data, str):
        rtstruct_dicom_data = pydicom.dcmread(rtstruct_dicom_data)

    logger.info(f"Loading {len(roi_index_map)} ROIs...")
    n_rois = len(roi_index_map)
    mask = ref_image_info.get_empty_numpy_array(VolumeOrder, np.int8)
    count = 1

    reverse_lookup = {}
    for roi_name, index in roi_index_map.items():
        reverse_lookup[index]=roi_name
        logger.info(f"Loading {roi_name}")
        contour = Contour.from_rtstruct(ref_image_info, rtstruct_dicom_data, roi_name)
        curr, m_info = contour.get_mask()
        if mask[curr].max() > 0:
            region = mask[curr]
            region = region[region>0]
            overlapping_label_indices = np.unique(region)
            labels = []
            for index in overlapping_label_indices:
                labels.append(reverse_lookup[index])
            logger.warning(f"Overlapping contours for {roi_name} with contours: {labels}")
        print(f"Adding {roi_name} with label index: {index}, voxels: {np.sum(curr.astype(np.int8))}")
        mask[curr] = index
        if progress_callback:
            progress_callback(int(count/n_rois*100))
            count+=1
    end = time.perf_counter()
    logger.info(f"Load took {end-start} seconds.")
    print(f"Unique data in mask: {np.unique(mask)}")
    return mask


class Contour:
    """
    This is a 3D volume contour class, a slice contour can be on xy, yz, or xz plane,
    Currently defaults to xy plane...

    If you plan on using other planes please write some tests in test_contour.py as this
    has only been tested on xy plane data

    """

    DEBUG = False
    TESTING_MASK = False
    BSPLINE_S = 0
    BSPLINE_FACTOR = 4  # number of points for each contour point (e.g. if 100 points in contour bspline will be 400)
    EPSILON = 1e-10

    logger = logging.getLogger("Contour")
    slice_mask_creator = None

    def __init__(self, name: str, contours: Union[List[NDArray], List[List[dict]]], image_info: VolumeInformation, contour_slice_normal=VolumeAxis.Z):
        """
        Initializes the Contour object.

        Args:
            name (str): The name of the contour (e.g., ROI name).
            contours (List[NDArray[np.float_]]): A list of NumPy arrays. Each array represents
                a contour slice and should have a shape of (N, 3), where N is the number
                of points in the contour slice, and the 3 columns represent (x, y, z)
                coordinates in physical units.
                Also support old format [[{'x':1,'y':1,'z':1},..{}],[{'x':1,'y':1,'z':1},..{}]]
                list of list of dicts
            image_info (VolumeInformation): Information about the associated image volume.
        """

        self._contour_slice_normal = contour_slice_normal
        # if you had your contours stored in ZYX no  problem handle it here
        self._contour_indices = VolumeIndices(VolumeIndexType.XYZ)

        if contours is not None and len(contours) > 0:
            if not isinstance(contours, list):
                raise TypeError("Contours must be a list")
            else:
                if isinstance(contours[0], list):
                    if isinstance(contours[0][0], dict):
                        contours = self.reformat_old_contours(contours)
                    else:
                        raise TypeError("Contours must either be a list of NumPy arrays size N,3 or a list of list of dicts")
                elif isinstance(contours[0], np.ndarray):
                    if contours[0].shape[1] != 3:
                        raise TypeError("Contours must be a list of NumPy arrays size N,3")
                else:
                    raise TypeError("Contours must either be a list of NumPy arrays size N,3 or a list of list of dicts")

        self._logger = logging.getLogger(name)
        self._image_info = image_info
        self._name = name
        if self._contour_slice_normal != VolumeAxis.Z:
            self._logger.warning(
                "You are using this class with contours on a slice other than XY, this is not well tested."
                " I would highly recommend testing the get_mask function especially with scaling"
            )

        # old way
        # self._contours = sorted(contours, key=lambda c: c[0]['z']) if contours else None  # sort contours

        self._contours = self.sort_contour_along_axis(contours, self._contour_slice_normal)
        self._slice_interp_method = itkMorphologicalMaskInterpolator()

    @property
    def name(self):
        return self._name

    @property
    def image_info(self):
        return self._image_info

    def set_slice_interp_method(self, interpolator: VolumeResize):
        assert isinstance(interpolator, VolumeResize)
        self._slice_interp_method = interpolator

    def sort_contour_along_axis(self, contours: List[NDArray], axis: VolumeAxis):
        if not contours:
            self._logger.warning("No contours found")
            return []

        axis_index = self._contour_indices[axis.value]

        # make sure each contour set has the same value in dimension normal to the slice
        for arr in contours:
            first_col_j = arr[:, axis_index]
            assert np.array_equal(arr[:, axis_index], first_col_j), f"Axis {axis} values are not the same for contour"

        sorted_array_list = sorted(contours, key=lambda arr: arr[0, axis_index])
        return sorted_array_list

    def reformat_old_contours(self, contours: List[List[dict]]) -> List[NDArray]:
        """
        Convert [[{'x':1,'y':2,'z':3},{'x':1,'y':2,'z':3}]] -> [np.array([[1,2,3],[1,2,3]])]
        :param contours:
        :return:
        """
        new_contours = []
        xyz_index = VolumeIndices(VolumeIndexType.XYZ)
        indexing = xyz_index.to(self._contour_indices.type)

        for c in contours:
            N = len(c)
            array = np.zeros([N, 3])
            for i, point in enumerate(c):
                array[i, :] = [point["x"], point["y"], point["z"]]

            new_contours.append(array[:, indexing])
        return new_contours

    @classmethod
    @property
    def mask_creator_types(cls):
        return list(cls.slice_mask_creator.keys())

    """
    @property
    def sorted_contours(self):
        return self._contours
    """

    @property
    def contours(self):
        return self._contours

    def get_physical_extent(self):
        """
        :return: the extent of the  roi in physical coordinates (x,y,z)
        """
        if not self._contours:
            self._logger.error("No contour to sort.")
        all_contours = np.concatenate(self._contours, axis=0)
        min = np.min(all_contours, axis=0)
        max = np.max(all_contours, axis=0)
        if self._contour_indices.type != VolumeIndexType.XYZ:
            min = min[self._contour_indices.to(VolumeIndexType.XYZ)]
            max = max[self._contour_indices.to(VolumeIndexType.XYZ)]
        return min, max

    def estimate_volume(self):
        # todo quick way to get a rois volume
        assert False

    @staticmethod
    def get_chunks(sequence, chunk_size):
        result = []
        for item in sequence:
            result.append(item)
            if len(result) >= chunk_size:
                yield result
                result = []
        if result:
            yield result  # yield the last, incomplete, portion

    @staticmethod
    def read_dicom_rois(rt_struct: Union[str, pydicom.dataset.FileDataset]) -> [str]:
        """
        Reads a RT-Struct file and returns the names of ROIs that are not POINT or MARKER types.
        :param rt_struct: can be either path to rtstruct file or loaded dataset
        :return:
        """
        return get_rtstruct_contour_roi_names(rt_struct)

    @classmethod
    # TODO: optimize RTSTRUCT reading, or pass in directly
    def from_rtstruct(cls, ref_image_info: VolumeInformation, rtstruct_dicom_data: Union[str, pydicom.dataset.FileDataset], roi_name: str) -> "Contour":
        """

        :param ref_image_info: reference series image information (size, origin, etc)
        :param rtstruct_dicom_data:  either file path to dicom or loaded pydicom data
        :param roi_name: name of contour to load
        :return:
        """

        cls.logger.debug(f"Loading {roi_name} contours from DICOM") #: {rtstruct_dicom_data}")
        if isinstance(rtstruct_dicom_data, str):
            rtstruct_dicom_data = pydicom.dcmread(rtstruct_dicom_data)

        assert isinstance(rtstruct_dicom_data, pydicom.dataset.FileDataset)

        roi_ss = [s for s in rtstruct_dicom_data.StructureSetROISequence if s.ROIName == roi_name]
        if len(roi_ss) != 1:
            raise Exception(f"Unable to find ROI ({roi_name}) in structure set: {rtstruct_dicom_data}")

        struct_set_roi = roi_ss[0]

        roi_cs = [s for s in rtstruct_dicom_data.ROIContourSequence if s.ReferencedROINumber == struct_set_roi.ROINumber]
        roi_contour = roi_cs[0]

        if any([c.ContourGeometricType != "CLOSED_PLANAR" for c in roi_contour.ContourSequence]):
            raise Exception("All slices are not CLOSED_PLANAR")

        # convert to same unit as image informat
        f = UnitConverter.get_conversion_factor(Unit.mm, ref_image_info.units)

        contour_data = [np.array([[p[0].real * f, p[1].real * f, p[2].real * f] for p in cls.get_chunks(c.ContourData, 3)]) for c in roi_contour.ContourSequence]

        contours = cls(**{"name": roi_name, "contours": contour_data, "image_info": ref_image_info})

        return contours

    @staticmethod
    def get_contour_in_pixels(
        contour: Union[list[tuple], NDArray], image_info: VolumeInformation, contour_normal=VolumeAxis.Z, contour_indices=VolumeIndices(VolumeIndexType.XYZ)
    ) -> (Union[NDArray, List[Tuple]], list):
        """
        Takes a contour (3d/2d) and returns the 2d points on the plane of interest
        :param contour: Could be 2d or 3d contour this is in true coordinates of image
        :param image_info: Image Information for contour
        :param contour_normal: this is the normal axis of the plane of the contour (e.g. for xy contour plane, it's Z)

        :return:
        """

        normal_index = contour_indices.get_index(contour_normal)

        # get the indices of the contour_plane
        plane_indices = [idx for idx in range(3) if idx != normal_index]

        if isinstance(contour, list):
            contour = np.array(contour)

        if contour.shape[1] == 2:
            contour_xyz = np.zeros((contour.shape[0], 3))
            contour_xyz[:, plane_indices] = contour
        else:
            contour_xyz = contour

        assert contour_xyz.shape[1] == 3

        # potentially this could change e.g. allow contours to be store YZX or something but
        # image_info expects data in xyz format
        assert image_info.image_information_index_type == contour_indices.type
        xyz_pixels = image_info.physical_point_to_continuous_index(contour_xyz)
        return xyz_pixels, plane_indices

    @staticmethod
    def get_contour_in_2d_pixels(
        contour: Union[list[tuple], NDArray],
        image_info: VolumeInformation,
        contour_normal=VolumeAxis.Z,
        contour_indices=VolumeIndices(VolumeIndexType.XYZ),
        return_tuple_list=False,
    ) -> Union[NDArray, List[Tuple]]:
        """
        Takes a contour (3d/2d) and returns the 2d points on the plane of interest
        :param contour_indices:
        :param contour: Could be 2d or 3d contour this is in true coordinates of image
        :param image_info: Image Information for contour
        :param contour_normal: this is the normal axis of the plane of the contour (e.g. for xy contour plane, it's Z)
        :param return_tuple_list: true if you want to return the data as a list of tuples otherwise you'll get a NDArray
        :return:
        """
        xyz, plane_indices = Contour.get_contour_in_pixels(contour, image_info, contour_normal, contour_indices)

        # for now we assume
        assert image_info.image_information_index_type == contour_indices.type

        normal_axis_index = contour_indices.get_index(contour_normal)
        if Contour.DEBUG:  # efficiency
            assert np.allclose(xyz[:, normal_axis_index], xyz[0, normal_axis_index])

        rtn_contour_pixels = xyz[:, plane_indices]
        if return_tuple_list:
            rtn_contour_pixels = [(x[0], x[1]) for x in rtn_contour_pixels]

        return rtn_contour_pixels

    # todo the set of functions get_slice_mask_from_*** would be better organized as a set of classes
    # e.g. SliceMaskGenerator with child classes SliceMaskPilGen, SliceMaskOpenCVGen etc...
    @staticmethod
    def get_slice_mask_from_pil(xy_contour: Union[list[tuple], NDArray], image_info: VolumeInformation) -> NDArray[bool]:
        """
        Uses PIL imagedra to create a mask image
        :param xy_contour: coordinates of contour assumed  to conform to 0,0 is center of pixel
        :param image_info: Information about the image to draw the contour on
        :return: boolean numpy array

        """
        # RS corner is the true corner of the corner grid voxel.
        # MIRA (DICOM and ITK) corner is the center of the corner grid voxel.

        # MIRA image indices are (0,0) at the centre of the corner grid voxel.
        # Pillow image indices are (0,0) at the corner of the corner grid voxel.
        # Therefore, MIRA and Pil indices do not coincide, and a half-voxel shift must be applied.
        if isinstance(xy_contour, list):
            xy_contour = [(x + 0.5, y + 0.5) for x, y in xy_contour]
        elif isinstance(xy_contour, np.ndarray):
            xy_contour = [tuple(row + 0.5) for row in xy_contour]
        else:
            raise Exception(f"Unsupported type {type(xy_contour)}")
        contour_image = Image.new("L", (int(image_info.nx), int(image_info.ny)))
        ImageDraw.Draw(contour_image).polygon(xy_contour, 1, outline=None)
        return np.array(contour_image).astype(bool)

    @staticmethod
    def get_slice_mask_from_opencv(xy_contour: Union[list[tuple], NDArray], image_info: VolumeInformation) -> NDArray[bool]:
        """
                Uses opencv to create a mask image, similar to pil results but it uses the "banker's/engineering rounding"
                :param xy_contour: coordinates of contour assumed  to conform to 0,0 is center of pixel and in
        c        continuous pixel coordinates
                :param image_info: Information about the image to draw the contour on
                :return: boolean numpy array

        """
        if isinstance(xy_contour, list):
            xy_contour_np = np.zeros([len(xy_contour), 1, 2], dtype=np.int32)
            for i, xy in enumerate(xy_contour):
                xy_contour_np[i, 0, :] = np.floor(np.array(xy) + 0.5).astype(np.int32)  # used to be round, but this results in 1.5 =2, and 2.5 =2
        elif isinstance(xy_contour, np.ndarray):
            xy_contour_np = (np.floor(xy_contour + 0.5).astype(np.int32)).reshape(xy_contour.shape[0], 1, 2)

        blank_im = np.zeros([image_info.ny, image_info.nx])
        contour_image = cv2.drawContours(blank_im, [xy_contour_np], -1, (1), thickness=cv2.FILLED)
        return contour_image.astype(bool)

    @staticmethod
    def get_slice_mask_from_mpl(xy_contour: Union[list[tuple], NDArray], image_info: VolumeInformation) -> NDArray[bool]:
        """
                This is uses Matplotlib Path, it results always in a smaller volume as it only considers pixels inside
                the contour path
                :param xy_contour: coordinates of contour assumed  to conform to 0,0 is center of pixel and in
        c        continuous pixel coordinates
                :param image_info: Information about the image to draw the contour on
                :return: boolean numpy array

        """
        negative_radius = True
        x, y = np.meshgrid(np.arange(int(image_info.nx)), np.arange(int(image_info.ny)))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        if not negative_radius:
            xy_contour = [(x - Contour.EPSILON, y - Contour.EPSILON) for x, y in xy_contour]
        path = Path(xy_contour)
        if negative_radius:
            grid = path.contains_points(points, radius=-Contour.EPSILON)  # Negative radius excludes boundary
        else:
            grid = path.contains_points(points)
        grid = grid.reshape((image_info.ny, image_info.nx))
        return np.array(grid).astype(bool)

    @staticmethod
    def get_slice_mask_from_rasterio(xy_contour: Union[list[tuple], NDArray], image_info: VolumeInformation) -> NDArray[bool]:
        transform = rasterio.transform.Affine(1, 0, -0.5, 0, 1, -0.5)
        """
        if we didn't have pixels in continuous pixel index coordinates
        we would do something like this:
        transom = raterio.transform.Affine(
            image_info.spacing[0],0, image_info.origin[0]-image_info.spacing[0]/2,
            0,image_info.spacing[1], image_info.origin[1]-image_info.spacing[1]/2,
        )

        """

        mask = geometry_mask([Polygon(xy_contour)], out_shape=(image_info.ny, image_info.nx), transform=transform, invert=True)
        return mask

    @staticmethod
    def get_slice_mask_from_skimage(xy_contour: Union[list[tuple], NDArray], image_info: VolumeInformation) -> NDArray[bool]:
        # Create an empty floating-point mask
        mask = np.zeros((image_info.ny, image_info.nx), dtype=bool)

        # Separate the x (column) and y (row) coordinates
        if isinstance(xy_contour, list):
            # The zip(*) idiom is a clean way to transpose a list of tuples
            c_coords, r_coords = zip(*xy_contour)
            r_coords = np.array(r_coords)
            c_coords = np.array(c_coords)
        elif isinstance(xy_contour, np.ndarray):
            r_coords = xy_contour[:, 1]
            c_coords = xy_contour[:, 0]
        else:
            raise TypeError(f"Unsupported type for xy_contour: {type(xy_contour)}")

        # Get the coordinates of pixels that are fully inside the polygon
        # These pixels will have a value of 1.0
        rr_in, cc_in = polygon(r_coords, c_coords, shape=mask.shape)
        mask[rr_in, cc_in] = True

        #todo we can do partial volumes with polygon

        # Get the coordinates and fractional values for the anti-aliased boundary
        # These are the pixels that the contour line intersects
        # rr_aa, cc_aa, val = polygon_aa(r_coords, c_coords, shape=mask.shape)
        # mask[rr_aa, cc_aa] = val

        return mask

    @staticmethod
    def get_slice_mask(xy_contour: list[tuple], image_info: VolumeInformation, type="pil") -> NDArray[bool]:
        """
        Returns a numpy boolean array of the same size as the image_info with True values where the contour intersects.
        :param type: 'PIL', 'opencv', 'mpl', or 'rasterio'
        :param xy_contour:
        :param image_info:
        :return:
        """
        creator = Contour.slice_mask_creator.get(type)
        if creator is None:
            raise ValueError(f"Unknown type for get_slice_mask {type}")
        return creator(xy_contour, image_info)

    def get_mask_image(self, type="pil", scale=None) -> sitk.Image:
        mask_array, info = self.get_mask(type, scale)
        return info.get_sitk_image(mask_array.astype(np.int8))

    def get_mask(
        self, type: str = "pil", scale: Union[list[int, int, int], np.ndarray] = None, crop: Union[list[int, int, int], np.ndarray] = None
    ) -> (np.array, VolumeInformation):
        """
        If you set scaling the in plane scaling is done using the contours by sampling the contours
        on a larger image size, the inter-slice interpolation can be set in set_slice_interpolator
        defaults to a morphological interpolator
        :param type: type of slice mask creation type to use
        :param scale: integer array or list for resizing image
        :param crop: None keeps the mask same physical extent as reference image, crop=[0,0,0] crops the image to just
        the contours extent, crop=[Nx,Ny,Nz] crops with a padding of Nx, Ny and Nz on each side of the array

        :return:
        """
        mask_image_info = self._image_info
        if crop is not None:
            extent_min, extent_max = self.get_physical_extent()
            size_before = mask_image_info.size
            mask_image_info.crop_info_to_physical_extent(extent_min, extent_max, crop)
            self._logger.debug(
                f"Cropping mask to contour, Size=( [X,Y,Z], Voxels) Pre=({size_before},{size_before.prod()}, Post={mask_image_info.size, mask_image_info.get_num_voxels()}"
            )

        # mask_array before z interpolation
        if self._contour_slice_normal == VolumeAxis.Z:
            mask_indices = VolumeIndices(VolumeIndexType.YXZ)
        elif self._contour_slice_normal == VolumeAxis.Y:
            mask_indices = VolumeIndices(VolumeIndexType.ZXY)
        elif self._contour_slice_normal == VolumeAxis.X:
            mask_indices = VolumeIndices(VolumeIndexType.ZYX)
        else:
            raise Exception("Unknown contour slice normal")

        sitk_normal_index = SITK_IMAGE_INDICES.get_index(self._contour_slice_normal)
        sitk_plane_indices = [idx for idx in range(3) if idx != sitk_normal_index]
        # scale in slice plane first
        if scale is not None:
            assert len(scale) == 3
            np_scale = np.array(scale)
            tmp_scale = np.ones(3)

            if scale[sitk_plane_indices[0]] != 1 or scale[sitk_plane_indices[1]] != 1:
                tmp_scale[sitk_plane_indices] = np_scale[sitk_plane_indices]
                mask_image_info = mask_image_info.get_resized_image_info(tmp_scale)

        mask_array = mask_image_info.get_empty_numpy_array(mask_indices.type, data_type=bool)

        mask_func = self.slice_mask_creator.get(type)
        if mask_func is None:
            raise ValueError(f"Unknown mask creation type: {type}")

        # todo use multiprocess to do each contour in it's on process
        for contour in self._contours:
            # RS corner is the true corner of the corner grid voxel.

            if len(contour) < 3:  # Need at least 3 points to create a closed contour
                print("Less than 3 points in contour. Skipping contour during mask generation")
                continue

            # RS corner is the true corner of the corner grid voxel.
            # MIRA (DICOM and ITK) corner is the center of the corner grid voxel.
            # MIRA image indices are (0,0) at the centre of the corner grid voxel.

            contour_pixels, plane_indices = self.get_contour_in_pixels(contour, mask_image_info, self._contour_slice_normal, self._contour_indices)
            contour_2d = contour_pixels[:, plane_indices]  # x,y or y,z or x,z

            # assume the slice is really perpendicular to slice normal for speed, test only in debug mode
            contour_normal_axis_index = self._contour_indices.get_index(self._contour_slice_normal)

            if Contour.DEBUG:
                assert np.allclose(contour_pixels[:, contour_normal_axis_index], contour_pixels[0, contour_normal_axis_index])

            slice_index = np.floor(contour_pixels[0, contour_normal_axis_index] + 0.5).astype(int)

            # any adjustment of voxel spacing is handled inside mask creation function

            slice_mask = mask_func(contour_2d, mask_image_info)

            # support contours being bigger than volume
            assert mask_indices.get_index(self._contour_slice_normal) == 2
            if slice_index >= mask_array.shape[mask_indices.get_index(self._contour_slice_normal)]:
                print(f"Contour is outside volume: ({slice_index}, {mask_array.shape[mask_indices.get_index(self._contour_slice_normal)]})")
            else:
                mask_array[:, :, slice_index] = np.bitwise_xor(mask_array[:, :, slice_index], slice_mask)

        # Change ordering from MASK is Y, X, Z  or
        # i.e. from PIL.Image to SimpleITK ordering
        # mask_array = np.moveaxis(mask_array, [0, 1, 2], [1, 2, 0])   Z,Y,X
        # old_order_mask = mask_array
        mask_array = np.moveaxis(mask_array, [0, 1, 2], mask_indices.move_to(SITK_ARRAY_TYPE))

        if scale is not None:
            slice_index = SITK_ARRAY_INDICES.get_index(self._contour_slice_normal)
            # make sure we didn't mess up and scale something we shouldn't have

            assert mask_array.shape[slice_index] == self._image_info.size[sitk_normal_index]
            assert scale[sitk_normal_index] >= 1
            assert isinstance(scale[sitk_normal_index], (int, np.integer))
            if scale[sitk_normal_index] > 1:
                mask_array, mask_image_info = self._slice_interp_method.resize_numpy_1d(
                    scale[sitk_normal_index], mask_array, mask_image_info, self._contour_slice_normal, SITK_ARRAY_TYPE
                )

        return mask_array, mask_image_info

    def _interp_contour(self, contour, interp_settings=None):
        if interp_settings is None:
            interp_settings = {"type": "splprep", "params": {"s": self.BSPLINE_S, "per": True}}
        c_np = np.array(contour)
        appended = False
        if not np.array_equal(c_np[0], c_np[-1]):
            c_np = np.vstack([c_np, c_np[0]])
            appended = True

        tck, u = splprep([c_np[:, 0], c_np[:, 1]], s=self.BSPLINE_S, per=True)  # Adjust "s" for smoothness
        u_new = np.linspace(0, 1, len(contour) * self.BSPLINE_FACTOR)  # More points for smoothness
        x_new, y_new = splev(u_new, tck)
        smooth_contour = np.array([x_new, y_new]).T
        if appended:
            smooth_contour = smooth_contour[:-1, :]
        return smooth_contour.tolist()


# because sometimes python syntax is shit... and won't let you reference functions inside a class definition
Contour.slice_mask_creator = {
    "pil": Contour.get_slice_mask_from_pil,
    "opencv": Contour.get_slice_mask_from_opencv,
    "mpl": Contour.get_slice_mask_from_mpl,
    "raster": Contour.get_slice_mask_from_rasterio,
    "skimage": Contour.get_slice_mask_from_skimage
}


if __name__ == "__main__":
    base = 5
    height = 5
    top_corner_xy = 3
    image_info = VolumeInformation([0, 0, 0], [1, 1, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1], [12, 12, 12])
    spacing = image_info.spacing
    first_pixel = top_corner_xy
    start = (first_pixel - spacing[0] / 2, first_pixel + spacing[1] / 2)

    triangle_xy = [
        start,
        (start[0], start[1] + height),
        (start[0] + base, start[1] + height),
        (start[0] + base, start[1]),
        # start
    ]
    triangle_xy

    mask = Contour.get_slice_mask_from_opencv(triangle_xy, image_info)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2)
    ax[0].imshow(mask)
    print("test")
