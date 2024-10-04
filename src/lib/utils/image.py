import cv2
import numpy as np
import torch


def get3rdPoint(a, b):
    """
    Computes the third point of a triangle.

    Args:
        a: The first point of the triangle.
        b: The second point of the triangle.

    Returns:
        The third point of the triangle.
    """
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def getDirection(src_point, rotateRadian):
    """
    Computes the direction of the rotated point.

    Args:
        src_point: The point to be rotated.
        rotateRadian: The rotation factor in radians.

    Returns:
        The direction vector of the rotated point.
    """
    sin = np.sin(rotateRadian)
    cos = np.cos(rotateRadian)
    return np.array(
        [
            src_point[0] * cos - src_point[1] * sin,
            src_point[0] * sin + src_point[1] * cos,
        ],
        dtype=np.float32,
    )


def getAffineTransform(
    center, scaleFactor, rotateFactor, outputSize, shift=(0, 0), inverse=False
):
    """
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
    """
    if not isinstance(scaleFactor, (np.ndarray, list)):
        scaleFactor = np.array([scaleFactor, scaleFactor], dtype=np.float32)

    src_w = scaleFactor[0]
    dst_w, dst_h = outputSize[0], outputSize[1]

    rotateRadian = np.pi * rotateFactor / 180.0
    srcDirection = getDirection([0, src_w * -0.5], rotateRadian)
    dstDirection = np.array([0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scaleFactor * shift
    src[1, :] = center + srcDirection + scaleFactor * shift
    dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst[1, :] = dstDirection + dst[0, :]

    src[2:, :] = get3rdPoint(src[0, :], src[1, :])
    dst[2:, :] = get3rdPoint(dst[0, :], dst[1, :])

    if inverse:
        return cv2.getAffineTransform(dst, src)
    else:
        return cv2.getAffineTransform(src, dst)


def affineTransform(points, transformMat):
    """
    Applies an affine transformation to a set of points.

    Args:
        point: The point to be transformed. (N, 2)
        transformMat: The affine transformation matrix.

    Returns:
        The transformed points.
    """
    if isinstance(points, torch.Tensor):
        matmul = torch.mm
        transformMat = torch.from_numpy(transformMat).to(points.device)
        newPoints = torch.ones(
            (points.shape[0], 3), dtype=torch.float32, device=points.device
        )
    else:
        matmul = np.dot
        newPoints = np.ones((points.shape[0], 3), dtype=np.float32)

    newPoints[:, :2] = points
    newPoints = matmul(transformMat, newPoints.T).T
    return newPoints[:, :2]


def lightingAug(image):
    """
    Applies lighting augmentation to an image.

    Args:
        image: The image to be augmented.

    Returns:
        The augmented image.
    """
    eig_val = torch.tensor(
        [0.2141788, 0.01817699, 0.00341571], device=image.device, requires_grad=False
    ).unsqueeze(1)
    eig_vec = torch.tensor(
        [
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938],
        ],
        device=image.device,
        requires_grad=False,
    )

    alpha = torch.empty(
        (3, 1), dtype=torch.float32, device=image.device, requires_grad=False
    )
    for i in range(3):
        alpha[i] = np.random.normal()
    alpha *= 0.1
    image += torch.mm(eig_vec, eig_val * alpha).unsqueeze(2)
    return image


def getGaussianRadius(det_size, min_overlap=0.7):
    """
    Computes the radius of the Gaussian kernel.

    Args:
        det_size: The size of the detection.
        min_overlap: The minimum overlap between the detection and the
            Gaussian kernel.

    Returns:
        The radius of the Gaussian kernel.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def getGaussianMatrix(shape, sigma=1.0):
    """
    Generate a 2D Gaussian kernel matrix.

    Args:
        shape (tuple): The shape of the desired Gaussian kernel matrix (rows, columns).
        sigma (float): Standard deviation of the Gaussian distribution. Default is 1.

    Returns:
        ndarray: A 2D NumPy array representing the Gaussian kernel matrix.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def ellip_gaussian2D(shape, sigma_x, sigma_y):
    """
    Code from https://github.com/zhangyp15/MonoFlex/blob/main/model/heatmap_coder.py#L83
    Generate a 2D Gaussian kernel matrix.

    Args:
        shape (tuple): The shape of the desired Gaussian kernel matrix (rows, columns).
        sigma_x (float): Standard deviation of the Gaussian distribution in the x-direction.
        sigma_y (float): Standard deviation of the Gaussian distribution in the y-direction.

    Returns:
        ndarray: A 2D NumPy array representing the Gaussian kernel matrix.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    # generate meshgrid
    h = np.exp(-(x * x) / (2 * sigma_x * sigma_x) - (y * y) / (2 * sigma_y * sigma_y))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h


def drawGaussianHeatRegion(heatmap, center, radius, k=1):
    """
    Draws a 2D Gaussian kernel on a heatmap. Inplace operation.

    Args:
        heatmap: The heatmap to be drawn on.
        center: The center of the Gaussian kernel.
        radius: The radius of the Gaussian kernel.
        k: The scaling factor of the Gaussian kernel.

    Returns:
        The heatmap with the Gaussian kernel drawn on it.
    """
    if isinstance(radius, int):
        diameter = 2 * radius + 1
        gaussian = getGaussianMatrix((diameter, diameter), sigma=diameter / 6)
        radius = [radius, radius]
    elif isinstance(radius, (tuple, list)):
        diameter_x, diameter_y = 2 * radius[0] + 1, 2 * radius[1] + 1
        gaussian = ellip_gaussian2D(
            (diameter_y, diameter_x), sigma_x=diameter_x / 6, sigma_y=diameter_y / 6
        )

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[:2]

    left, right = min(x, radius[0]), min(width - x, radius[0] + 1)
    top, bottom = min(y, radius[1]), min(height - y, radius[1] + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius[1] - top : radius[1] + bottom, radius[0] - left : radius[0] + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap
