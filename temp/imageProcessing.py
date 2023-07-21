import numpy as np
import cv2
import random
from scipy.ndimage import rotate

class DataAugmentation():
    @staticmethod
    def translate(origin_img, shift=10, direction=None, roll=True):
        assert direction in ['right', 'left', 'down', 'up', None], 'Directions should be top|up|left|right'
        if shift==0:
            return
            
        if direction is None:
            directions = ['right', 'left', 'down', 'up']
            direction = directions[random.randint(0, 3)]

        img = origin_img.copy()
        shift = random.randint(1, shift)
        if direction == 'right':
            right_slice = img[:, -shift:].copy()
            img[:, shift:] = img[:, :-shift]
            if roll:
                img[:,:shift] = np.fliplr(right_slice)
        elif direction == 'left':
            left_slice = img[:, :shift].copy()
            img[:, :-shift] = img[:, shift:]
            if roll:
                img[:, -shift:] = left_slice
        elif direction == 'down':
            down_slice = img[-shift:, :].copy()
            img[shift:, :] = img[:-shift,:]
            if roll:
                img[:shift, :] = down_slice
        elif direction == 'up':
            upper_slice = img[:shift, :].copy()
            img[:-shift, :] = img[shift:, :]
            if roll:
                img[-shift:,:] = upper_slice
        origin_img[:] = img

    @staticmethod
    def random_crop(origin_img, crop_factor:float):
        origin_size = origin_img.shape
        crop_size = (int(origin_size[1] * crop_factor), int(origin_size[0] * crop_factor))
        img = origin_img.copy()
        w, h = img.shape[:2]
        x, y = np.random.randint(h-crop_size[0]), np.random.randint(w-crop_size[1])
        img = img[y:y+crop_size[0], x:x+crop_size[1]]
        origin_img[:] = cv2.resize(img, origin_size[:2]).reshape(origin_size)

    @staticmethod
    def rotate_img(origin_img, angle, bg_patch=(5,5)):
        assert len(origin_img.shape) <= 3, "Incorrect image shape"
        rgb = len(origin_img.shape) == 3
        img = origin_img.copy()
        if rgb:
            bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
        else:
            bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])

        angle = random.randint(-angle, angle)
        img = rotate(img, angle, reshape=False)
        mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
        img[mask] = bg_color
        origin_img[:] = img

    @staticmethod
    def gaussian_noise(origin_img, mean=0, sigma=0.03):
        img = origin_img.copy()
        noise = np.random.normal(mean, sigma, img.shape)
        mask_overflow_upper = img + noise >= 1.0
        mask_overflow_lower = img + noise < 0
        noise[mask_overflow_upper] = 1.0
        noise[mask_overflow_lower] = 0
        img += noise
        origin_img[:] = img

    @staticmethod
    def brightness(img, factor=0.8):
        img[:] = img[:] * (factor + np.random.uniform(0, 0.4)) #scale channel V uniformly
        img[:][img > 1] = 1 #reset out of range values
        
    @staticmethod
    def contrast(img, factor=0.8):
        mean = np.mean(img)
        img[:] = (img - mean) * (factor + np.random.uniform(0, 0.4)) + mean


# Unit test
if __name__ == '__main__':
    import os
    import time
    startTime = time.time()
    print("Start imageProcessing testing...")
    DataAug = DataAugmentation()

    if not os.path.exists('unitTest'):
        os.mkdir('unitTest')

    # Read test images
    colorLenna = cv2.imread('Lenna_(test_image).png') / 255
    grayLenna = cv2.imread('Lenna_(test_image).png', 0) / 255

    # Test grayscale images
    grayLennas = [grayLenna.copy() for _ in range(6)]
    AugNames = ['translate', 'random_crop', 'rotate_img', 'gaussian_noise', 'brightness', 'contrast']
    DataAug.translate(grayLennas[0], shift=50)
    DataAug.random_crop(grayLennas[1], crop_factor=0.8)
    DataAug.rotate_img(grayLennas[2], 10)
    DataAug.gaussian_noise(grayLennas[3])
    DataAug.brightness(grayLennas[4], factor=0.5)
    DataAug.contrast(grayLennas[5], factor=0.5)
    for i, name in enumerate(AugNames):
        grayLennas[i] = (grayLennas[i] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join('unitTest', f'{name}Gray.jpg'), grayLennas[i])

    # Test color images
    colorLennas = [colorLenna.copy() for _ in range(6)]
    DataAug.translate(colorLennas[0], shift=50)
    DataAug.random_crop(colorLennas[1], crop_factor=0.8)
    DataAug.rotate_img(colorLennas[2], 10)
    DataAug.gaussian_noise(colorLennas[3])
    DataAug.brightness(colorLennas[4], factor=0.5)
    DataAug.contrast(colorLennas[5], factor=0.5)
    for i, name in enumerate(AugNames):
        colorLennas[i] = (colorLennas[i] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join('unitTest', f'{name}Color.jpg'), colorLennas[i])

    print(f"Time used: {time.time() - startTime:.2f} secs")