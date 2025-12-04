import datetime
import glob
import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def find_series_uid_for_roi(dicom_path: str, rtstruct_file: Union[str, pydicom.dataset.FileDataset], roi_name):
    logger = logging.getLogger(__name__)
    sop_uids = get_sop_uids_for_roi(rtstruct_file, roi_name)
    dcm_files = glob.glob(os.path.join(dicom_path, "*.dcm"))

    for dcm_file in dcm_files:
        try:
            ds = pydicom.dcmread(dcm_file)
            if ds.SOPInstanceUID in sop_uids:
                return getattr(ds, "SeriesInstanceUID", None)
        except Exception as e:
            logger.error(f"Error reading file {dcm_file}: {e}")

    return None


def get_ref_image_series_uid(rtstruct_file):
    logger = logging.getLogger(__name__)
    pyds = rtstruct_file
    if isinstance(pyds, str):
        pyds = pydicom.dcmread(pyds)
    elif not isinstance(pyds, pydicom.dataset.Dataset):
        raise TypeError("rtstruct_file must be a path to a DICOM RTStruct file or a pydicom dataset.")

        # Get the outer sequence safely
    ref_frames = getattr(pyds, 'ReferencedFrameOfReferenceSequence', [])

    for frame in ref_frames:
        # Safely get studies, default to empty list if missing
        studies = getattr(frame, 'RTReferencedStudySequence', [])

        for study in studies:
            # Safely get series, default to empty list if missing
            series_seq = getattr(study, 'RTReferencedSeriesSequence', [])

            for series in series_seq:
                # We found a series, return the UID immediately
                if hasattr(series, 'SeriesInstanceUID'):

                    return series.SeriesInstanceUID

    logger.warning("Could not find any Reference Image Series UID in the provided structure.")
    return None


def get_sop_uids_for_roi(rtstruct_file, roi_name):
    ds = rtstruct_file
    if isinstance(ds, str):
        ds = pydicom.dcmread(ds)
    elif not isinstance(ds, pydicom.dataset.FileDataset):
        raise TypeError("rtstruct_file must be a path to a DICOM RTStruct file or a pydicom dataset.")

    for roi in ds.StructureSetROISequence:
        if roi.ROIName == roi_name:
            roi_number = roi.ROINumber
            break

    if roi_number is None:
        raise ValueError(f"ROI Name '{roi_name}' not found in the RTSTRUCT file.")

    sop_instance_uids = []
    for roi_contour in ds.ROIContourSequence:
        if roi_contour.ReferencedROINumber == roi_number:
            for contour_sequence in roi_contour.ContourSequence:
                contour_images = contour_sequence.ContourImageSequence
                sop_instance_uids.extend(image.ReferencedSOPInstanceUID for image in contour_images)

    return sop_instance_uids


def get_rtstruct_roi_names(rtstruct_file: Union[str, pydicom.dataset.FileDataset]):
    if isinstance(rtstruct_file, str):
        rtstruct_file = pydicom.dcmread(rtstruct_file)
    elif not isinstance(rtstruct_file, pydicom.dataset.FileDataset):
        raise TypeError("rtstruct_file must be a path to a DICOM RTStruct file or a pydicom dataset.")
    roi_names = [roi.ROIName for roi in rtstruct_file.StructureSetROISequence]
    return roi_names


def get_rtstruct_contour_roi_names(rtstruct_file: Union[str, pydicom.dataset.FileDataset]):
    if isinstance(rtstruct_file, str):
        rtstruct_file = pydicom.dcmread(rtstruct_file)
    elif not isinstance(rtstruct_file, pydicom.dataset.FileDataset):
        raise TypeError("rtstruct_file must be a path to a DICOM RTStruct file or a pydicom dataset.")
    names = {r.ROINumber: r.ROIName for r in rtstruct_file.StructureSetROISequence}
    geo_types = {r.ReferencedROINumber: r.ContourSequence[0].ContourGeometricType for r in rtstruct_file.ROIContourSequence}

    rois = []
    for roi_num, geo_type in geo_types.items():
        if geo_type not in ["POINT", "MARKER"]:
            rois.append(names[roi_num])
    return rois

def get_unique_series_uids(root_directory):
    """
    Scans a directory for DICOM files and returns a list of unique
    Series Instance UIDs.

    Args:
        root_directory (str): The path to the directory to search.

    Returns:
        list: A list of unique SeriesInstanceUID strings.
    """
    unique_uids = set()
    search_path = Path(root_directory)

    # Check if directory exists
    if not search_path.is_dir():
        raise FileNotFoundError(f"The directory {root_directory} does not exist.")

    print(f"Scanning '{root_directory}' for DICOM series...")

    # rglob('*') recursively finds all files in the directory
    for file_path in search_path.rglob('*'):
        if file_path.is_file():
            try:
                # stop_before_pixels=True ensures we only read the header
                # This makes the process significantly faster
                dcm_data = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)

                # Check if SeriesInstanceUID exists in the header
                if 'SeriesInstanceUID' in dcm_data:
                    unique_uids.add(dcm_data.SeriesInstanceUID)

            except ( TypeError, OSError):
                # Skip files that are not valid DICOMs (e.g., .txt, .jpg, or corrupted files)
                continue

    return list(unique_uids)


def find_all_uids(dicom_file: Union[str, pydicom.dataset.FileDataset]):
    if isinstance(dicom_file, str):
        ds = pydicom.dcmread(dicom_file)
    elif isinstance(dicom_file, pydicom.dataset.FileDataset):
        ds = dicom_file
    else:
        raise TypeError("dicom_file must be a path to a DICOM RTStruct file or a pydicom dataset.")
    uids = []

    # Recursive function to extract UIDs
    def extract_uids(dataset):
        for elem in dataset:
            if elem.VR == "UI":  # VR = "UI" indicates a UID
                uids.append((elem.tag, elem.name, elem.value))
            elif elem.VR == "SQ":  # Handle sequences
                for item in elem.value:
                    extract_uids(item)

    extract_uids(ds)
    return uids


def write_dicom_from_sitk(sitk_image, output_directory, series_instance_uid, additional_metadata=None, patient_name="Anon", modality="CT", patient_id="000001"):
    """
    Write a SimpleITK image to DICOM format with a specific Series Instance UID using PyDICOM

    Parameters:
    sitk_image (sitk.Image): The SimpleITK image to write
    output_directory (str): Directory where to save the DICOM files
    series_instance_uid (str): The specific Series Instance UID to use
    additional_metadata (dict, optional): Additional metadata to include in DICOM files
    """
    orientation_mat = np.reshape(sitk_image.GetDirection(), (3, 3))
    assert np.allclose(np.linalg.norm(orientation_mat, axis=0), 1)

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Convert SimpleITK image to numpy array (axes may be flipped)
    # SimpleITK uses x,y,z indexing, numpy uses z,y,x
    image_array = sitk.GetArrayFromImage(sitk_image)

    # Extract SimpleITK metadata
    sitk_metadata = {}
    for key in sitk_image.GetMetaDataKeys():
        sitk_metadata[key] = sitk_image.GetMetaData(key)

    # Extract image spacing, origin, and direction
    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    direction = sitk_image.GetDirection()

    # Generate UID for the study if not found in metadata
    study_instance_uid = None
    if "0020|000d" in sitk_metadata:
        study_instance_uid = sitk_metadata["0020|000d"]
    else:
        study_instance_uid = generate_uid()

    # Get current date and time for DICOM metadata
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    current_time = datetime.datetime.now().strftime("%H%M%S")

    # Check if image is 2D or 3D
    is_3d = len(image_array.shape) == 3

    # Set up basic file meta
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # For 3D volumes, write each slice separately
    if not is_3d:
        image_array.resize(1, image_array.shape[0], image_array.shape[1])

    num_slices = image_array.shape[0]
    for i in range(num_slices):
        # Extract slice
        slice_data = image_array[i, :, :]

        # Create unique SOP Instance UID for this slice
        sop_instance_uid = generate_uid()
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid

        # Create the dataset with the slice data
        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Populate required DICOM tags
        ds.SOPInstanceUID = sop_instance_uid
        ds.SeriesInstanceUID = series_instance_uid
        ds.StudyInstanceUID = study_instance_uid
        ds.StudyDate = current_date
        ds.StudyTime = current_time
        ds.ContentDate = current_date
        ds.ContentTime = current_time
        ds.Modality = modality  # Default, override if in metadata
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.InstanceNumber = i + 1

        # Apply spacing information
        # Note: SimpleITK and DICOM have different coordinate conventions
        ds.PixelSpacing = [spacing[1], spacing[0]]  # y, x
        if is_3d:
            ds.SliceThickness = spacing[2]

        # Set slice position information (using DICOM coordinate system)
        ds.ImagePositionPatient = (
            [origin[0] + i * spacing[2] * direction[2], origin[1] + i * spacing[2] * direction[5], origin[2] + i * spacing[2] * direction[8]]
            if is_3d
            else [origin[0], origin[1], origin[2]]
        )

        # Image Orientation (first two direction cosines of the image)
        ds.ImageOrientationPatient = [direction[0], direction[1], direction[2], direction[3], direction[4], direction[5]]

        # Add image pixel data
        if slice_data.dtype != np.uint16:
            # Rescale to uint16 range if needed
            if np.max(slice_data) > 0:
                rescaled = ((slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 65535).astype(np.uint16)
            else:
                rescaled = slice_data.astype(np.uint16)
            slice_data = rescaled

        ds.Rows, ds.Columns = slice_data.shape
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0  # unsigned
        ds.PixelData = slice_data.tobytes()
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        # Transfer SimpleITK metadata to DICOM dataset
        for tag, value in sitk_metadata.items():
            # Skip UIDs that we're explicitly setting
            if tag in ["0020|000d", "0020|000e"]:
                continue

            try:
                group, element = tag.split("|")
                pydicom_tag = pydicom.tag.Tag((int(group, 16), int(element, 16)))
                ds[pydicom_tag] = value
            except Exception as e:
                print(f"Warning: Could not set metadata {tag}: {e}")

        # Add any additional metadata if provided
        if additional_metadata:
            for key, value in additional_metadata.items():
                if hasattr(ds, key):
                    setattr(ds, key, value)

        # Save the file
        if is_3d:
            output_filename = os.path.join(output_directory, f"slice_{i:03d}.dcm")
        else:
            output_filename = os.path.join(output_directory, "image.dcm")
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.save_as(output_filename, write_like_original=False)

    print(f"DICOM {'series' if is_3d else 'image'} saved with Series Instance UID: {series_instance_uid}")


def get_series_instance_uids(directory):
    logger = logging.getLogger(__name__)
    series_uids = set()  # Use a set to avoid duplicates
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                if "SeriesInstanceUID" in ds:
                    series_uids.add(ds.SeriesInstanceUID)
            except Exception as e:
                logger.error(f"Could not process file {file_path}: {e}")

    return series_uids


def get_series_instance_uids_from_dicom_dir(dicom_path):
    reader = sitk.ImageSeriesReader()
    return reader.GetGDCMSeriesIDs(dicom_path)


def load_dicom_images(dicom_dir_path, series_instance_uid=None) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir_path, series_instance_uid)

    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir_path}")

    reader.SetFileNames(dicom_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    return reader.Execute()


def load_dicom_image(dicom_file_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(dicom_file_path)
    image = reader.Execute()
    return image



def get_series_image_info(dicom_directory: str, series_uid: str, slice_thickness_check: bool = False, single_series_check: bool = False) -> dict:
    """
    Retrieves DICOM series information (dimensions, spacing, direction, origin)
    without loading all image data.

    Args:
        dicom_directory: The directory containing the DICOM files.
        series_uid: The Series Instance UID of the series to retrieve information from, if None assumes single series and performs
        a check to ensure data directory contains only one series uid
        slice_thickness_check: checks all slices have the same thickness throws a exception if they don't, add computational cost
        as every series needs to be opened
        single_series_check: checks to ensure that there truly is only one series in the directory, will
        throw exception if not
    Returns:
        A dictionary containing the series information, or None if the series
        is not found or an error occurs.
    """
    logger = logging.getLogger(__name__)
    series_info = None
    reader = sitk.ImageSeriesReader()

    if series_uid is None and single_series_check:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_directory)
        if len(series_ids) > 1:
            msg = (
                f"Series UID was set to None, assumes single series in dir: {dicom_directory} but found these uids: {series_ids}\n"
                f"Turn off single_series_check to just use first series and not check. "
            )
            logger.error(msg)
            raise ValueError(msg)
        series_uid = series_ids[0]

    if series_uid is not None:
        dicom_files = reader.GetGDCMSeriesFileNames(dicom_directory, series_uid)
    else:
        dicom_files = reader.GetGDCMSeriesFileNames(dicom_directory)

    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found for series UID {series_uid} in {dicom_directory}")

    if series_uid is None and slice_thickness_check:
        logger.warning("You are assuming single series uid for directory and will compare all series images for slice thickness complaince irregardless of uid")
    thicknesses = []

    for filepath in dicom_files:
        try:
            ds = pydicom.dcmread(filepath, stop_before_pixels=True)

            if ds.SeriesInstanceUID == series_uid or series_uid is None:
                series_info = {
                    "Rows": ds.Rows,
                    "Columns": ds.Columns,
                    "NumberOfFrames": getattr(ds, "NumberOfFrames", 1),  # Handle single-frame
                    "PixelSpacing": ds.PixelSpacing,
                    "ImageOrientationPatient": ds.ImageOrientationPatient,
                    "ImagePositionPatient": ds.ImagePositionPatient,
                    "SliceThickness": getattr(ds, "SliceThickness", None),  # Might not always be present
                    "FrameOfReferenceUID": getattr(ds, "FrameOfReferenceUID", None),
                }

                if series_info["SliceThickness"] is None:
                    raise AttributeError(f"No SliceThickness found in DICOM file for series {filepath}")

                thicknesses.append(series_info["SliceThickness"])

                if not slice_thickness_check:
                    break
                if thicknesses[-1] != thicknesses[0]:
                    raise ValueError(f"Slice thicknesses are not consistent across all slices in series {series_uid}.")


        except (pydicom.errors.InvalidDicomError, FileNotFoundError) as e:
            logger.error(f"Error reading DICOM file '{filepath}': {e}")

    if series_info is None:
        raise ValueError(f"Series with UID '{series_uid}' not found in directory '{dicom_directory}'")

    orientation = np.append(series_info["ImageOrientationPatient"], np.cross(series_info["ImageOrientationPatient"][:3], series_info["ImageOrientationPatient"][3:]))
    volume_info = {
        "size": np.array([series_info["Columns"], series_info["Rows"], len(dicom_files)]),
        "spacing": np.array([series_info["PixelSpacing"][1], series_info["PixelSpacing"][0], series_info["SliceThickness"]]),
        "origin": np.array(series_info["ImagePositionPatient"]),
        "orientation": orientation,
        "frame_of_reference": series_info["FrameOfReferenceUID"],
    }

    return volume_info


# This just uses sitk using C++ library
def get_series_image_info_sitk(dicom_directory: str, series_uid: str, slice_thickness_check: bool = False, single_series_check: bool = False) -> dict:
    """
    Retrieves DICOM series information (dimensions, spacing, direction, origin)
    without loading all image data.
    """
    logger = logging.getLogger(__name__)
    series_info = None
    series_reader = sitk.ImageSeriesReader()

    if series_uid is None and single_series_check:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_directory)
        if len(series_ids) > 1:
            msg = (
                f"Series UID was set to None, assumes single series in dir: {dicom_directory} but found these uids: {series_ids}\n"
                f"Turn off single_series_check to just use first series and not check."
            )
            logger.error(msg)
            raise ValueError(msg)
        series_uid = series_ids[0]

    if series_uid is not None:
        dicom_files = series_reader.GetGDCMSeriesFileNames(dicom_directory, series_uid)
    else:
        dicom_files = series_reader.GetGDCMSeriesFileNames(dicom_directory)

    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found for series UID {series_uid} in {dicom_directory}")

    if series_uid is None and slice_thickness_check:
        logger.warning("You are assuming single series uid for directory and will compare all series images for slice thickness compliance regardless of uid.")

    thicknesses = []

    for filepath in dicom_files:
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(filepath)
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()

            file_series_uid = reader.GetMetaData("0020|000e")  # SeriesInstanceUID
            if file_series_uid == series_uid or series_uid is None:
                series_info = {
                    "Rows": int(reader.GetMetaData("0028|0010")),
                    "Columns": int(reader.GetMetaData("0028|0011")),
                    "PixelSpacing": [float(x) for x in reader.GetMetaData("0028|0030").split("\\")],
                    "ImageOrientationPatient": [float(x) for x in reader.GetMetaData("0020|0037").split("\\")],
                    "ImagePositionPatient": [float(x) for x in reader.GetMetaData("0020|0032").split("\\")],
                    "SliceThickness": float(reader.GetMetaData("0018|0050")) if reader.HasMetaDataKey("0018|0050") else None,
                    "FrameOfReferenceUID": reader.GetMetaData("0020|0052") if reader.HasMetaDataKey("0020|0052") else None,
                }

                if series_info["SliceThickness"] is None:
                    raise AttributeError(f"No SliceThickness found in DICOM file for series {filepath}")

                thicknesses.append(series_info["SliceThickness"])

                if not slice_thickness_check or thicknesses[-1] == thicknesses[0]:
                    break

        except Exception as e:
            logger.error(f"Error reading DICOM file '{filepath}': {e}")

    if series_info is None:
        raise ValueError(f"Series with UID '{series_uid}' not found in directory '{dicom_directory}'")

    orientation = np.append(series_info["ImageOrientationPatient"], np.cross(series_info["ImageOrientationPatient"][:3], series_info["ImageOrientationPatient"][3:]))
    volume_info = {
        "size": np.array([series_info["Columns"], series_info["Rows"], len(dicom_files)]),
        "spacing": np.array([series_info["PixelSpacing"][1], series_info["PixelSpacing"][0], series_info["SliceThickness"]]),
        "origin": np.array(series_info["ImagePositionPatient"]),
        "orientation": orientation,
        "frame_of_reference": series_info["FrameOfReferenceUID"],
    }

    return volume_info
