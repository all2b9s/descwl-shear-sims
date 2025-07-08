# Run Sims func
import numpy as np
# use galaxy models from WeakLensingDeblending.  Note you need
# to get the data for this, see below for downloading instructions
from descwl_shear_sims.galaxies import WLDeblendGalaxyCatalog
# The star catalog class
from descwl_shear_sims.stars import StarCatalog
from descwl_shear_sims.sim import make_sim
# for making a power spectrum PSF
from descwl_shear_sims.psfs import make_ps_psf
# convert coadd dims to SE dims, need for this PSF
from descwl_shear_sims.sim import get_se_dim

def run_sim_DC2(rng, coadd_dim, buff, bands = ['i'], shear = [0,0],
                rotate_galaxy = True,
                psf_model = 'Gauss',
                psf_fwhm = None,
                noise_factor = 1,
                draw_noise = True,
                shear_psf = True,
                galaxy_nfac = 1.0,
                select_observable = ['i_ab', 'a_d'],
                select_upper_limit = [24,100],
                select_lower_limit = [0,0.2]):
    """
    Function to generate simulated data for meta-detection.

    Args:
        rng (RandomState): 
            Random number generator for reproducibility.
        coadd_dim (int): 
            Dimension of the coadd image.
        buff (int): 
            Buffer size around the coadd image.
        bands (list): 
            List of bands to simulate.
        shear (list of float): 
            Shear values for the simulated data.

    Returns:
        sim_data (dict): 
            dict of simulated Exposures.
    """
    pixel_scale = 0.2
    rotate = False
    dither = True
    
    # this is the single epoch image sized used by the sim, we need
    # it for the power spectrum psf
    se_dim = get_se_dim(coadd_dim=coadd_dim, coadd_scale=0.2, se_scale = 0.2, rotate=rotate)
    # galaxy catalog;
    galaxy_catalog = WLDeblendGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        pixel_scale=pixel_scale,
        select_observable = select_observable,
        select_upper_limit = select_upper_limit,
        select_lower_limit = select_lower_limit,
        do_rotate=rotate_galaxy,
        n_factor=galaxy_nfac,
    )
    # star catalog; 
    '''star_catalog = StarCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        density = 10,
    )'''

    psf = make_ps_psf(rng=rng, dim=se_dim, model = psf_model)
    if psf_fwhm is not None:
        psf.force_psf_width = psf_fwhm
    psf.shear_psf = shear_psf
    sim_data = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        #star_catalog=star_catalog,
        coadd_dim=coadd_dim,
        g1=shear[0],
        g2=shear[1],
        psf=psf,
        dither=dither,
        rotate=rotate,
        bands=bands,
        noise_factor= noise_factor,
        sky_n_sigma = 0,
        cosmic_rays=False,
        bad_columns=False,
        star_bleeds=False,
        draw_noise=draw_noise,
    )
    return sim_data, se_dim, galaxy_catalog
