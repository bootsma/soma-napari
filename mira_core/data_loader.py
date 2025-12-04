import os.path

from volume_info import VolumeInformation
from dicom_utils import find_all_uids, find_series_uid_for_roi, get_rtstruct_roi_names, get_ref_image_series_uid, get_unique_series_uids


def load_image_and_mask( rt_struct_path: str, dicom_image_set_path:str):
    return VolumeInformation.get_volume_info_and_array(rt_struct_path, dicom_image_set_path)




if __name__ == "__main__":


    test_rt_struct = os.path.abspath(r'..\test-data')
    test_dir = os.path.join(test_rt_struct, 'Fx3')
    dicomrt_data_path = os.path.join(test_dir, 'structures', 'RS1_correct.dcm')
    image_dir = os.path.join(test_dir,'ImageSet')

    rois = get_rtstruct_roi_names(dicomrt_data_path)
    print("rois: ", rois)
    result = find_series_uid_for_roi(image_dir, dicomrt_data_path, rois[0])
    print(f"ROI: {rois[0]} : {result}")
    result2=  get_ref_image_series_uid(dicomrt_data_path)
    print(f"DICOM: {result}")

    unique_series_uids = get_unique_series_uids(image_dir)
    print(f'UIDS in DIR: {unique_series_uids}')
    try:
        vol_info, vol = load_image_and_mask(dicomrt_data_path, image_dir)
        print(vol_info)


    except Exception as e:
        print("Failed to load data:", e)



    test_rt_struct = os.path.abspath(r'..\test-data')
    test_dir = os.path.join(test_rt_struct, 'Fx2')
    dicomrt_data_path = os.path.join(test_dir, 'structures', 'RS1_correct.dcm')
    image_dir = os.path.join(test_dir,'ImageSet')

    rois = get_rtstruct_roi_names(dicomrt_data_path)
    print("rois: ", rois)
    result = find_series_uid_for_roi(image_dir, dicomrt_data_path, rois[0])
    print(f"ROI: {rois[0]} : {result}")
    result2=  get_ref_image_series_uid(dicomrt_data_path)
    print(f"DICOM: {result}")

    unique_series_uids = get_unique_series_uids(image_dir)
    print(f'UIDS in DIR: {unique_series_uids}')
    try:
        vol_info, vol = load_image_and_mask(dicomrt_data_path, image_dir)
        print(vol_info)


    except Exception as e:
        print("Failed to load data:", e)



    test_rt_struct = os.path.abspath(r'..\test-data')
    test_dir = os.path.join(test_rt_struct, 'A')
    dicomrt_data_path = os.path.join(test_dir,  'RS1_correct.dcm')
    image_dir = test_dir


    rois = get_rtstruct_roi_names(dicomrt_data_path)
    print("rois: ", rois)
    result = find_series_uid_for_roi(image_dir, dicomrt_data_path, rois[0])
    print(f"ROI: {rois[0]} : {result}")
    result2=  get_ref_image_series_uid(dicomrt_data_path)
    print(f"DICOM: {result}")

    try:
        vol_info, vol = load_image_and_mask(dicomrt_data_path, image_dir)
        print(vol_info)


    except Exception as e:
        print("Failed to load data:", e)





