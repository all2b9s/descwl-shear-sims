import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import cv2
import galsim
from mdet.geom_tool import shear_pos

def classic_single_galaxy(args):
    d, method = args
    pred_shape = []
    truth_pos = []
    if d is None:
        print("Empty datadict")
        return None
    hdul = fits.open(d["filename"]+'_i.fits')
    img = hdul[1].data
    psf = hdul[2].data
    filename_parts = d["filename"].split('/')
    img_head = filename_parts[-1]
    folder = '/'.join(filename_parts[:-1])
    
    

    galsim_psf = galsim.Image(psf, scale=0.2)
    for obj in d['annotations']:
        try:
            x, y, w, h = (i for i in obj['bbox'])
            x_sheared, y_sheared = shear_pos(x, y, img_head, img_shape = img.shape)
            x_sheared = int(x_sheared)
            y_sheared = int(y_sheared)
            cutout = (img)[y_sheared:y_sheared+h, x_sheared:x_sheared+w]
            galsim_img = galsim.Image(cutout, scale=0.2)
            if method == 'Regauss':
                result = galsim.hsm.EstimateShear(galsim_img, galsim_psf)
            elif method == 'AdaptiveMom':
                result = galsim.hsm.FindAdaptiveMom(galsim_img)
            else:
                raise ValueError(f"Unknown method: {method}")
            pred_shape.append([result.observed_shape.e1, result.observed_shape.e2])
            truth_pos.append(shear_pos(x+w//2, y+h//2, img_head, img_shape = img.shape))
            #truth_shape.append([obj['et_1'], obj['et_2']])
        except Exception as e:
            #print(f"Error processing object in image {d['filename']}: {e}")
            pass
    pred_shape = np.array(pred_shape)
    truth_pos = np.array(truth_pos)
    temp = np.concatenate((truth_pos, pred_shape), axis=1)
    filename_parts = d["filename"].split('/')
    img_head = filename_parts[-1]
    folder = '/'.join(filename_parts[:-1])
    np.save(f'{folder}/{method}_{img_head}_measured_shape.npy', temp)
    return temp
