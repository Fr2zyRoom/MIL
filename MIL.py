import numpy as np
import pydicom


def cxr_loader(dcm_path):
    """load CXR image(DICOM)
    
    Parameters:
        dcm_path (str) -- a dicom file path
    """
    dcm_info = pydicom.read_file(dcm_path, force=True)
    dcm_img = dcm_info.pixel_array
    if dcm_info.get("RescaleIntercept"):
        intercept = dcm_info.RescaleIntercept
        slope = dcm_info.RescaleSlope
    
        dcm_img = dcm_img.astype(np.float64) * slope + intercept
    
    dcm_img = pydicom.pixel_data_handlers.apply_voi_lut(dcm_img, dcm_info)
    
    #complement
    if dcm_info[0x0028, 0x0004].value != "MONOCHROME2" :
        norm_img = 1 - (dcm_img - np.min(dcm_img)) / (np.max(dcm_img) - np.min(dcm_img))
    else:
        norm_img = (dcm_img - np.min(dcm_img)) / (np.max(dcm_img) - np.min(dcm_img))
    
    return np.uint8(norm_img*255)
