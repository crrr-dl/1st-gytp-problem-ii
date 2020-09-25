import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def random_elastic(img, label = None):
    
    alpha = np.array(img.shape) * 2
    sigma = np.array(img.shape) * (np.random.rand(3) * 0.02 + 0.04)

    dx = gaussian_filter((np.random.rand(*img.shape) * 2 - 1), sigma[0]) * alpha[0]
    dy = gaussian_filter((np.random.rand(*img.shape) * 2 - 1), sigma[1]) * alpha[1]
    dz = gaussian_filter((np.random.rand(*img.shape) * 2 - 1), sigma[2]) * alpha[2]

    x, y, z = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), np.arange(img.shape[2]), indexing = 'ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))]

    if label is None:
        return map_coordinates(img, indices, order = 0, mode = 'nearest').reshape(img.shape)
    else:
        return map_coordinates(img, indices, order = 0, mode = 'nearest').reshape(img.shape), map_coordinates(label, indices, order = 0, mode = 'nearest').reshape(label.shape)


def add_gaussian_noise(img):
    
    sigma = np.random.rand() * 0.1
    gauss = sigma * np.random.randn(*img.shape).astype(np.float32)
    return img + gauss