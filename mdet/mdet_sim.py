# Mdet Func
import ngmix
import galsim
import lsst.geom as geom
import numpy as np

# This function performs metacalibration on a simulated image.
def mdet_sim(sim, se_dim, band, rng):
    """
    Perform metadetection on a simulated image.

    Parameters:
    sim (dict): 
        The dict for target image.
    se_dim (int): 
        The pixel number of the image.
    band (string): 
        The band.
    rng (numpy.random.Generator): 
        The random number generator.

    Returns:
    dict: A dictionary containing the metacalibration results.
    """

    band_data = sim['band_data'][band][0]
    wcs = sim['se_wcs'][band][0]
    # We use the psf at the center of the image
    psf_image = band_data.getPsf().computeKernelImage(geom.Point2D([se_dim//2, se_dim//2])).array
    psf_shape = psf_image.shape

    # Get the wcs jacobian at the center of the image
    gsjac = wcs.jacobian(galsim.PositionD(se_dim//2, se_dim//2))
    
    # Set up the ngmix jacobian for psf, using the coordinate in psf image
    ngmix_jac_psf = ngmix.Jacobian(
        x=psf_shape[0] // 2,
        y=psf_shape[1] // 2,
        dudx=gsjac.dudx,
        dudy=gsjac.dudy,
        dvdx=gsjac.dvdx,
        dvdy=gsjac.dvdy,
    )
    psf_observation = ngmix.Observation(psf_image, jacobian=ngmix_jac_psf)

    sim_image = band_data.getImage().getArray()
    sim_vari = band_data.getVariance().getArray()
    # Set up the ngmix jacobian, for sim image, using the coordinate in sim image
    ngmix_jac = ngmix.Jacobian(
        x=se_dim // 2,
        y=se_dim // 2,
        dudx=gsjac.dudx,
        dudy=gsjac.dudy,
        dvdx=gsjac.dvdx,
        dvdy=gsjac.dvdy,
    )
    observation = ngmix.Observation(sim_image, weight = 1/sim_vari, jacobian=ngmix_jac, psf=psf_observation)
    mdet_dict = ngmix.metacal.get_all_metacal(observation, psf='fitgauss', rng=rng, 
                                              types = ['noshear','1p','1m','2p','2m'])
    return mdet_dict, psf_image

