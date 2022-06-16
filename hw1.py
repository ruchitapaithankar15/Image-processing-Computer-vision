
"""
Imports we need.
Note: You may _NOT_ add any more imports than these.
"""
import argparse
from turtle import left
from unittest import result
import imageio
import logging
import numpy as np
from PIL import Image


def load_image(filename):
    """Loads the provided image file, and returns it as a numpy array."""
    im = Image.open(filename)
    return np.array(im)

def rgb2ycbcr(im):
    """Convert RGB to YCbCr."""
    xform = np.array([[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [0.5, -0.4187, -0.0813]], dtype=np.float32)
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128.
    return ycbcr.astype(np.uint8)

def ycbcr2rgb(im):
    """Convert YCbCr to RGB."""
    xform = np.array([[1., 0., 1.402], [1, -0.34414, -0.71414], [1., 1.772, 0.]], dtype=np.float32)
    rgb = im.astype(np.float32)
    rgb[:,:,[1,2]] -= 128.
    rgb = rgb.dot(xform.T)
    return np.clip(rgb, 0., 255.).astype(np.uint8)

def create_gaussian_kernel(size, sigma=1.0):
    """
    Creates a 2-dimensional, size x size gaussian kernel.
    It is normalized such that the sum over all values = 1. 

    Args:
        size (int):     The dimensionality of the kernel. It should be odd.
        sigma (float):  The sigma value to use 

    Returns:
        A size x size floating point ndarray whose values are sampled from the multivariate gaussian.

    See:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        https://homepages.inf.ed.ac.uk/rbf/HIPR2/eqns/eqngaus2.gif
    """

    # Ensure the parameter passed is odd
    if size % 2 != 1:
        raise ValueError('The size of the kernel should not be even.')

    # Creates a size by size ndarray of type float32
    matrixSize = [size,size]
    rv = np.zeros(matrixSize, dtype=np.float32)
    i = int((size - 1) / 2)
    
    # Populates the values of the kernel. Note that the middle `pixel` should be x = 0 and y = 0.
    for x in range(-i, i +1):
        for y in range(-i, i+1):
            raiseTo = -((x ** 2 + y ** 2) / (2 * sigma ** 2 ))
            rv[x + i][y + i] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(raiseTo)

    #Normalizes the values such that the sum of the kernel = 1
    norm = sum(map(sum,rv))
    rv = rv/norm
    return rv


def convolve_pixel(img, kernel, i, j):
    """
    Convolves the provided kernel with the image at location i,j, and returns the result.
    If the kernel stretches beyond the border of the image, it returns the original pixel.

    Args:
        img:        A 2-dimensional ndarray input image.
        kernel:     A 2-dimensional kernel to convolve with the image.
        i (int):    The row location to do the convolution at.
        j (int):    The column location to process.

    Returns:
        The result of convolving the provided kernel with the image at location i, j.
    """

    # First let's validate the inputs are the shape we expect...
    if len(img.shape) != 2:
        raise ValueError(
            'Image argument to convolve_pixel should be one channel.')
    if len(kernel.shape) != 2:
        raise ValueError('The kernel should be two dimensional.')
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError(
            'The size of the kernel should not be even, but got shape %s' % (str(kernel.shape)))
    
    # determining using the kernel shape, the ith and jth locations to start at.
    k = int((kernel.shape[0]-1) / 2) 
    kernel = np.rot90(kernel,2) #rotating the kernel by 90 degree twice.
    outside = 0
    upper, lower = i - k, i + k
    left,right = j - k, j + k

    # Checking if the kernel stretches beyond the border of the image.
    if (upper < 0) or (lower >= img.shape[0]) or (left<0) or (right >= img.shape[1]):
        outside = 1

    #the kernel stretches beyond the border of the image - return the input pixel at that location.
    if outside:
        return img[i][j]

    #perfoms convolution
    else:
        pixelValues = []
        for x in range(-k, k+1):
            for y in range(-k, k+1):
                pixelValue = kernel[x + k][y + k] * img[i + x][j + y]
                pixelValues.append(pixelValue)

        result = np.sum(pixelValues) 
        return result
        

def convolve(img, kernel):
    """
    Convolves the provided kernel with the provided image and returns the results.

    Args:
        img:        A 2-dimensional ndarray input image.
        kernel:     A 2-dimensional kernel to convolve with the image.

    Returns:
        The result of convolving the provided kernel with the image at location i, j.
    """
    # Makes a copy of the input image to save results
    results = np.zeros(img.shape)

    # Populates each pixel in the input by calling convolve_pixel and return results.
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            populatePixel = convolve_pixel(img,kernel,x,y)
            results[x][y] = populatePixel
    rounded = np.round(results)
    results = np.array(rounded, dtype=np.uint8)
    return results


def split(img):
    """
    Splits a image (a height x width x 3 ndarray) into 3 ndarrays, 1 for each channel.

    Args:
        img:    A height x width x 3 channel ndarray.

    Returns:
        A 3-tuple of the r, g, and b channels.
    """
    if img.shape[2] != 3:
        raise ValueError('The split function requires a 3-channel input image')
    split_img = np.dsplit(img,3)
    for i in range(3):
        split_img[i] = np.squeeze(split_img[i])
    R,G,B = split_img
    return (R,G,B)




def merge(r, g, b):
    """
    Merges three images (height x width ndarrays) into a 3-channel color image ndarrays.

    Args:
        r:    A height x width ndarray of red pixel values.
        g:    A height x width ndarray of green pixel values.
        b:    A height x width ndarray of blue pixel values.

    Returns:
        A height x width x 3 ndarray representing the color image.
    """
    #merging into 3-channel color image ndarrays.
    merged = np.dstack((r,g,b))
    return merged


"""
The main function
"""
if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Blurs an image using an isotropic Gaussian kernel.')
    parser.add_argument('input', type=str, help='The input image file to blur')
    parser.add_argument('output', type=str, help='Where to save the result')
    parser.add_argument('--ycbcr', action='store_true', help='Filter in YCbCr space')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='The standard deviation to use for the Guassian kernel')
    parser.add_argument('--k', type=int, default=5,
                        help='The size of the kernel.')
    parser.add_argument('--subsample', type=int, default=1, help='Subsample by factor')

    args = parser.parse_args()

    # first load the input image
    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)

    if args.ycbcr:
        # Convert to YCbCr
        inputImage = rgb2ycbcr(inputImage)

        # Split it into three channels
        logging.info('Splitting it into 3 channels')
        (y, cb, cr) = split(inputImage)

        # compute the gaussian kernel
        logging.info('Computing a gaussian kernel with size %d and sigma %f' %
                     (args.k, args.sigma))
        kernel = create_gaussian_kernel(args.k, args.sigma)

        # convolve it with cb and cr
        logging.info('Convolving the Cb channel')
        cb = convolve(cb, kernel)
        logging.info('Convolving the Cr channel')
        cr = convolve(cr, kernel)

        # merge the channels back
        logging.info('Merging results')
        resultImage = merge(y, cb, cr)

        # convert to RGB
        resultImage = ycbcr2rgb(resultImage)
    else:
        # Split it into three channels
        logging.info('Splitting it into 3 channels')
        (r, g, b) = split(inputImage)

        # compute the gaussian kernel
        logging.info('Computing a gaussian kernel with size %d and sigma %f' %
                     (args.k, args.sigma))
        kernel = create_gaussian_kernel(args.k, args.sigma)

        # convolve it with each input channel
        logging.info('Convolving the first channel')
        r = convolve(r, kernel)
        logging.info('Convolving the second channel')
        g = convolve(g, kernel)
        logging.info('Convolving the third channel')
        b = convolve(b, kernel)

        # merge the channels back
        logging.info('Merging results')
        resultImage = merge(r, g, b)

    # subsample image
    if args.subsample != 1:
        # subsample by a factor of 2
        resultImage = resultImage[::args.subsample, ::args.subsample, :]

    # save the result
    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, resultImage)
