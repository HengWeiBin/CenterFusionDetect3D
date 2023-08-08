import cv2
import numpy as np
import torch
from torchvision.transforms import \
    ColorJitter, Normalize, Lambda, Compose, RandomOrder

def get3rdPoint(a, b):
    '''
    Computes the third point of a triangle.

    Args:
        a: The first point of the triangle.
        b: The second point of the triangle.

    Returns:
        The third point of the triangle.
    '''
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def getDirection(src_point, rotateRadian):
    '''
    Computes the direction of the rotated point.

    Args:
        src_point: The point to be rotated.
        rotateRadian: The rotation factor in radians.

    Returns:
        The direction vector of the rotated point.
    '''
    sin = np.sin(rotateRadian)
    cos = np.cos(rotateRadian)
    return np.array([src_point[0] * cos - src_point[1] * sin,
                     src_point[0] * sin + src_point[1] * cos],
                    dtype=np.float32)

def getAffineTransform(
        center, scaleFactor, rotateFactor, outputSize, shift=(0, 0), inverse=False):
    '''
    Computes an affine transformation matrix.

    Args:
        center: The center of the image to be scaled and rotated.
        scaleFactor: The scale factor of the image.
        rotateFactor: The rotation factor of the image in degrees.
        output_size: The size of the output image.
        shift: The translation of the image.
        inv: Whether to compute the inverse transformation matrix.

    Returns:
        The affine transformation matrix.
    '''
    if not isinstance(scaleFactor, (np.ndarray, list)):
        scaleFactor = np.array([scaleFactor, scaleFactor], dtype=np.float32)

    src_w = scaleFactor[0]
    dst_w = outputSize[0]
    dst_h = outputSize[1]

    rotateRadian = np.pi * rotateFactor / 180.0
    srcDirection = getDirection([0, src_w * -0.5], rotateRadian)
    dstDirection = np.array([0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scaleFactor * shift
    src[1, :] = center + srcDirection + scaleFactor * shift
    dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst[1, :] = dstDirection + np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)

    src[2:, :] = get3rdPoint(src[0, :], src[1, :])
    dst[2:, :] = get3rdPoint(dst[0, :], dst[1, :])

    if inverse:
        return cv2.getAffineTransform(dst, src)
    else:
        return cv2.getAffineTransform(src, dst)
    
def affineTransform(point, transformMat):
    '''
    Applies an affine transformation to a point.

    Args:
        point: The point to be transformed.
        transformMat: The affine transformation matrix.

    Returns:
        The transformed point.
    '''
    newPoint = np.array([point[0], point[1], 1.0], dtype=np.float32)
    newPoint = np.dot(transformMat, newPoint)
    return point[:2]
    
def lightingAug(image):
    '''
    Applies lighting augmentation to an image.

    Args:
        image: The image to be augmented.

    Returns:
        The augmented image.
    '''
    eig_val = torch.tensor([0.2141788, 0.01817699, 0.00341571],
                           device=image.device,
                           requires_grad=False).unsqueeze(1)
    eig_vec = torch.tensor([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]],
        device=image.device,
        requires_grad=False)

    alpha = torch.empty((3, 1),
                        dtype=torch.float32,
                        device=image.device,
                        requires_grad=False)
    for i in range(3):
        alpha[i] = np.random.normal()
    alpha *= 0.1
    image += torch.mm(eig_vec, eig_val * alpha).unsqueeze(2)
    return image