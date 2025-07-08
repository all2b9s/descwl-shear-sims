import numpy as np
import galsim
from galcheat.utilities import mag2counts, mean_sky_level
import btk
import cv2
import pandas as pd
from astropy.table import Table

def e1e2_to_ephi(e1,e2):
    
    pa = np.arctan(e2/e1)
    
    return pa

L0 = 3.0128e28

def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin-4, rmax+4, cmin-4, cmax+4

def dcut_reformat(cat, only_bttri=True):
    
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        cat[f'{band}_ab'] = cat[f'{band}_ab']
    
    # only bulge to total ratio i-band from the cosmoDC2 GCR Catalog
    if only_bttri:
        total_flux = L0 * 10**(-0.4*cat[f'i_ab'])
        bulge_to_total_ratio = cat[f'fluxnorm_bulge']

        cat[f'fluxnorm_bulge_i'] = total_flux * bulge_to_total_ratio
        cat[f'fluxnorm_disk_i'] = total_flux * (1-bulge_to_total_ratio)
        cat[f'fluxnorm_agn_i'] = np.zeros(total_flux.shape)
    else:
        for band in ['u', 'g', 'r', 'i', 'z', 'y']:
            total_flux = L0 * 10**(-0.4*cat[f'{band}_ab'])
            bulge_to_total_ratio = cat['fluxnorm_bulge']

            cat[f'fluxnorm_bulge_{band}'] = total_flux * bulge_to_total_ratio
            cat[f'fluxnorm_disk_{band}'] = total_flux * (1-bulge_to_total_ratio)
            cat[f'fluxnorm_agn_{band}'] = np.zeros(total_flux.shape)

    #cat['a_b'] = cat['size_bulge_true']
    #cat['b_b'] = cat['size_minor_bulge_true']

    #cat['a_d'] = cat['size_disk_true']
    #cat['b_d'] = cat['size_minor_disk_true']

    cat['pa_bulge'] = e1e2_to_ephi(cat['ellipticity_1_bulge_true'],cat['ellipticity_2_bulge_true']) * 180.0/np.pi

    cat['pa_disk'] = e1e2_to_ephi(cat['ellipticity_1_disk_true'],cat['ellipticity_2_disk_true']) * 180.0/np.pi
    
    cat['pa_tot'] = e1e2_to_ephi(cat['ellipticity_1_true'],cat['ellipticity_2_true']) * 180.0/np.pi

    cat['g1'] = cat['shear_1']
    cat['g2'] = cat['shear_2']
    
    return cat

def make_galaxy(entry, survey, filt, no_disk= False, no_bulge = False, no_agn = True):
    components = []
    total_flux = mag2counts(entry[filt + "_ab"], survey.name, filt).to_value("electron")
    # Calculate the flux of each component in detected electrons.
    total_fluxnorm = entry["fluxnorm_disk_"+filt] + entry["fluxnorm_bulge_"+filt] + entry["fluxnorm_agn_"+filt]
    disk_flux = 0.0 if no_disk else entry["fluxnorm_disk_"+filt] / total_fluxnorm * total_flux
    bulge_flux = 0.0 if no_bulge else entry["fluxnorm_bulge_"+filt] / total_fluxnorm * total_flux
    agn_flux = 0.0 if no_agn else entry["fluxnorm_agn_"+filt] / total_fluxnorm * total_flux

    if disk_flux + bulge_flux + agn_flux == 0:
        raise SourceNotVisible

    if disk_flux > 0:
        a_d, b_d = entry["a_d"], entry["b_d"]
        disk_hlr_arcsecs=a_d
        

        disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr_arcsecs).shear(
            e1=entry['ellipticity_1_disk_true'], e2=entry['ellipticity_2_disk_true']
        ).rotate(entry['sim_angle']* galsim.degrees)
        
        components.append(disk)
        
        
    if bulge_flux > 0:
        a_b, b_b = entry["a_b"], entry["b_b"]
        bulge_hlr_arcsecs = np.sqrt(a_b * b_b)
        
        bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs).shear(
           e1=entry['ellipticity_1_bulge_true'], e2=entry['ellipticity_2_bulge_true']
        ).rotate(entry['sim_angle']* galsim.degrees)
        components.append(bulge)

    if agn_flux > 0:
        agn = galsim.Gaussian(flux=agn_flux, sigma=1e-8)
        components.append(agn)

    profile = galsim.Add(components)
    return profile

def make_seg(entry, survey, filt, lvl,nx=128,ny=128):
    psf = survey.get_filter(filt).psf
    sky_level = mean_sky_level(survey, filt).to_value('electron') # gain = 1
    obj_type = entry['truth_type'] # 1 for galaxies, 2 for stars
    im = None
    if obj_type == 1:
        gal = make_galaxy(entry, survey, survey.get_filter(filt))
        #gal = gal.shear(g1=entry["g1"], g2=entry["g2"])
        conv_gal = galsim.Convolve(gal, psf)
        im = conv_gal.drawImage(
            nx=nx,
            ny=nx,
            scale=survey.pixel_scale.to_value("arcsec")
        )
    '''else:
        star, gsparams, isbright = make_star(entry, survey, survey.get_filter(filt))
        max_n_photons = 10_000_000
        # 0 means use the flux for n_photons 
        mag =entry['mag_'+filt]
        flux = mag2counts(mag,survey,filt).to_value("electron")
        n_photons = 0 if flux < max_n_photons else max_n_photons
        #n_photons = 0 if entry[f'flux_{filt}'] < max_n_photons else max_n_photons
        conv_star = galsim.Convolve(star, psf)
        im = conv_star.drawImage(
            nx=nx,
            ny=nx,
            scale=survey.pixel_scale.to_value("arcsec"),
            method="phot",
            n_photons=n_photons,
            poisson_flux=True,
            maxN=1_000_000,  # shoot in batches this size
            rng=grng
        )'''
        
    imd = np.expand_dims(np.expand_dims(im.array,0),0)
    # thresh for mask set relative to the bg noise level which is what sigma_noise is
    # so lower the thresh for the star to include more of its light
    # so lower sigma_noise, bigger masks and higher lvl, smaller masks bc it'll only capture very brightest central part of star
    if obj_type == 2: # if star, 
        segs = btk.metrics.utils.get_segmentation(imd, sky_level, sigma_noise=lvl)
    else:
        segs = btk.metrics.utils.get_segmentation(imd, sky_level, sigma_noise=lvl)        
    return segs[0][0]

def make_im(entry, survey, filt, lvl,nx=128,ny=128):
    psf = survey.get_filter(filt).psf
    sky_level = mean_sky_level(survey.name, filt).to_value('electron') # gain = 1
    obj_type = entry['truth_type'] # 1 for galaxies, 2 for stars
    im = None
    if obj_type == 1:
        gal = make_galaxy(entry, survey, filt)
        #gal = gal.shear(g1=entry["g1"], g2=entry["g2"])
        conv_gal = galsim.Convolve(gal, psf)
        im = conv_gal.drawImage(
            nx=nx,
            ny=nx,
            scale=survey.pixel_scale.to_value("arcsec")
        )
    '''else:
        star, gsparams, isbright = make_star(entry, survey, survey.get_filter(filt))
        max_n_photons = 10_000_000
        # 0 means use the flux for n_photons 
        #mag = -2.5*np.log10(entry[f'flux_{filt}']*1e-9/(1e23*10**(48.6/-2.5)))
        mag =entry['mag_'+str(filt)]
        flux = mag2counts(mag,survey,filt).to_value("electron")
        n_photons = 0 if flux < max_n_photons else max_n_photons
        #n_photons = 0 if entry[f'flux_{filt}'] < max_n_photons else max_n_photons
        conv_star = galsim.Convolve(star, psf)
        im = conv_star.drawImage(
            nx=nx,
            ny=nx,
            scale=survey.pixel_scale.to_value("arcsec"),
            method="phot",
            n_photons=n_photons,
            poisson_flux=True,
            maxN=1_000_000,  # shoot in batches this size
            rng=grng
        )'''
        
    return im

def create_metadata(img_shape, cat, survey, filt, lvl=3, star_cat = None):

    """ Code to format the metadatain to a dict.  It takes the i-band and makes a footprint+bounding boxes
    from thresholding to sn*sky_level
    
    Parameters
    
    blend_batch: BTK blend batch
        BTK batch of blends
    sky_level: float
        The background sky level in the i-band
    sn: int
        The signal-to-noise ratio for thresholding
    idx:
        The index of the blend in the blend_batch
        
    Returns
        ddict: dict
            The dictionary of metadata for the idx'th blend in the batch 
    
    """
    

    ddict = {}

    ddict[f"file_name"] = 'none'
    ddict["image_id"] = 0
    ddict["height"] = img_shape[0]
    ddict["width"] = img_shape[1]
    
    
    t = Table.from_pandas(cat)
    #t = cat

    n = len(cat)
    objs = []
    for j in range(n):

        obj = t[j]
        
        #a = math.ceil(obj['size_true']/0.2)*2
        #b = math.ceil(obj['size_minor_true']/0.2)*2
        x = obj['new_x']
        y = obj['new_y']
        #mask = make_seg(obj,survey,filt, lvl)
        
        segs = []
        for filt in ['u','g','r','i','z','y']:
            im  = make_im(obj, survey, filt, lvl=2, nx=128,ny=128)
    
            imd = np.expand_dims(np.expand_dims(im.array,0),0)
            sky_level = mean_sky_level(survey.name, filt).to_value('electron') # gain = 1
            segs.append(btk.metrics.utils.get_segmentation(imd, sky_level, sigma_noise=2))

        mask = np.clip(np.sum(segs,axis=0), a_min=0, a_max=1)[0][0]
        
        
        #mask=cv2.ellipse(frame, (frame.shape[0]//2,frame.shape[1]//2), (a,b), pa, 0 , 360, (255,0,0), -1)
        #frame = np.zeros((dat.shape[1],dat.shape[2]))
        #mask=cv2.ellipse(frame, (0,0), (a,b), pa, 0 , 360, (255,0,0), -1)
#         print(obj["truth_type"], ": " ,mask, "\n")
        if np.sum(mask)==0:
            continue
        
        bbox = get_bbox(mask)
        x0 = bbox[2]
        x1 = bbox[3]
        y0 = bbox[0]
        y1 = bbox[1]
        
        w = x1-x0
        h = y1-y0
        
        bbox = [x-w/2, y-h/2, w, h]     

        redshift = obj['redshift']
        obj_id = obj['object_id']
        mag_i = obj['i_ab']
        et_1 = obj['ellipticity_1_true']
        et_2 = obj['ellipticity_2_true']

        contours, hierarchy = cv2.findContours(
                    (mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )


        segmentation = []
        for contour in contours:
            # contour = [x1, y1, ..., xn, yn]
            contour = contour.flatten()
            if len(contour) > 4:
                contour[::2] += (int(np.rint(x))-x0-w//2)
                contour[1::2] += (int(np.rint(y))-y0-h//2)
                #contour[::2] += (int(y)-y0-h//2)
                #contour[1::2] += (int(x)-x0-w//2)
                
                segmentation.append(contour.tolist())
        # No valid countors
        if len(segmentation) == 0:
            print(j)
            continue

        obj = {
            "bbox": bbox,
            "area": w*h,
            #"bbox_mode": BoxMode.XYWH_ABS,
            "bbox_mode": 1,
            "segmentation": segmentation,
            "category_id": 1 if obj['truth_type'] == 2 else 0,
            "redshift": redshift,
            "obj_id": obj_id,
            "mag_i": mag_i,
            "et_1": et_1,
            "et_2": et_2,
        }
        objs.append(obj)
        
    
    ddict['annotations'] = objs

    return ddict