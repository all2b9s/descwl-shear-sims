import numpy as np

def shear_pos(x, y, mode, img_shape = [1070, 1070]):
    if mode == '1p':
        shear = np.array([0.01, 0.0])
    elif mode == '1m':
        shear = np.array([-0.01, 0.0])
    elif mode == '2p':
        shear = np.array([0.0, 0.01])
    elif mode == '2m':
        shear = np.array([0.0, -0.01])
    else:
        return x, y
    delta_x = x-img_shape[0]//2
    delta_y = y-img_shape[1]//2
    x_sheared = delta_x + shear[0] * delta_x + shear[1] * delta_y + img_shape[0]//2
    y_sheared = delta_y + shear[1] * delta_x - shear[0] * delta_y + img_shape[1]//2

    return x_sheared, y_sheared

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