import numpy as np
from astropy.io import fits
import pandas as pd
import shutil
import json
import os
import lsst.geom as geom

def rotate_ellipticity(e, angle):
    """Rotate the ellipticity e by the given angle in degrees."""
    z = complex(e[0], e[1])
    angle_rad = np.deg2rad(angle)
    z_rotated = z * np.exp(1j * angle_rad*2)
    return [z_rotated.real, z_rotated.imag]

def e2vec(e, angle = 0):
    z = complex(e[0], e[1])
    vec = np.sqrt(z)* np.exp(1j*angle/ 180 * np.pi)  # Convert angle from degrees to radians
    return np.array([vec.real, vec.imag])

def save_mdet(mdet_dict, sim_data, filepath, band):
    band_data = sim_data['band_data'][band][0]
    ori_image = band_data.getImage().getArray()
    se_dim = ori_image.shape[0]
    # We use the psf at the center of the image
    psf_image = band_data.getPsf().computeKernelImage(geom.Point2D([se_dim//2, se_dim//2])).array

    
    hdu_list = fits.HDUList()
    hdu_list.append(fits.PrimaryHDU())  # Add an empty primary HDU
    hdu_list.append(fits.ImageHDU(ori_image, name='original_image'))  # Add the original image HDU
    hdu_list.append(fits.ImageHDU(psf_image, name='psf_image'))  # Add the psf image HDU
    ori_filepath = filepath+'ori_'+band+'.fits'
    hdu_list.writeto(ori_filepath, overwrite=True)  # Save the original image as a separate file
    del hdu_list
    # Save the MDET images
    
    if mdet_dict is not None:
        for obs in mdet_dict:
            hdu_list = fits.HDUList()
            hdu_list.append(fits.PrimaryHDU()) 
            image = mdet_dict[obs].image
            psf = mdet_dict[obs].psf.image
            hdu_list.append(fits.ImageHDU(image, name=obs))
            hdu_list.append(fits.ImageHDU(psf, name=obs+'_psf'))
            obs_filepath = filepath + obs +'_'+ band + '.fits'
            try:
                os.remove(obs_filepath)
            except:
                pass
            hdu_list.writeto(obs_filepath, overwrite=True)
            #hdu_list.pop()  # Remove the last HDU to prepare for the next observation
            del hdu_list

def build_mdet_cat(sim_data, gal_cat, shear=[0, 0]):
    """
    Build the MDET catalog based on simulation data and galaxy catalog.

    Args:
        sim_data (dict): Simulation data.
        gal_cat (object): Galaxy catalog.
        shear (list, optional): Shear values. Defaults to [0, 0].

    Returns:
        DataFrame: MDET catalog.
    """
    
    sim_id = sim_data['truth_info']['index']
    sim_cat = gal_cat._wldeblend_cat[sim_id]
    sim_angle = gal_cat.angles
    sim_e = np.array([sim_cat['ellipticity_1_true'], sim_cat['ellipticity_2_true']]).transpose()
    sim_e = np.array([rotate_ellipticity(e, sim_angle[i]) for i, e in enumerate(sim_e)])  # Rotate ellipticities

    # Update the ellipticity in sim_cat
    sim_cat['ellipticity_1_true'] = sim_e[:, 0]+shear[0]
    sim_cat['ellipticity_2_true'] = sim_e[:, 1]+shear[1]
    
    # Convert structured array to DataFrame
    sim_df = pd.DataFrame(sim_cat)
    
    # Add the new columns
    sim_df['truth_type'] = 1
    sim_df['new_x'] = sim_data['truth_info']['image_x']
    sim_df['new_y'] = sim_data['truth_info']['image_y']
    sim_df['object_id'] = sim_data['truth_info']['index']
    sim_df['sim_angle'] = sim_angle 
    sim_df['shear_1'] = shear[0]
    sim_df['shear_2'] = shear[1]

    return sim_df


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def convert_to_json(dict_list, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dict_list: list of metadata dictionaries
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    print(f"Caching COCO format annotations at '{output_file}' ...")
    tmp_file = output_file + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(dict_list, f,cls=NpEncoder)
    shutil.move(tmp_file, output_file)