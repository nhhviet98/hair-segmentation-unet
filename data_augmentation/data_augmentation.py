import cv2
import numpy as np


class DataAugmentation:
    def __init__(self, input_size: int, output_size: int):
        """
        Init of data augmentation class
        :param input_size: int
            input size of image to use for Unet model
        :param output_size:
            output size of image to use for Unet model
        """
        self.input_size = input_size
        self.output_size = output_size

    def resize_image(self, image: np.ndarray, mask: np.ndarray):
        """
        Resize image to fit to input Unet model
        :param image: np.ndarray
            Input image
        :param mask: np.ndarray
            Input mask
        :return: np.ndarray
            image and mask after resized.
        """
        w = image.shape[1]
        h = image.shape[0]
        ratio = min(w, h)/max(w, h)
        if w >= h:
            new_w = self.input_size
            new_h = round(self.input_size*ratio)
        else:
            new_h = self.input_size
            new_w = round(self.input_size*ratio)
        image_resized = cv2.resize(image, (new_w, new_h))
        mask_resized = cv2.resize(mask, (new_w, new_h))
        return image_resized, mask_resized

    def padding_image(self, image: np.ndarray, mask: np.ndarray):
        """
        Padding image to fit to Unet model
        :param image: np.ndarray
            Input image
        :param mask: np.ndarray
            Input mask
        :return: np.ndarray
            Image and mask after padding
        """
        h = image.shape[0]
        w = image.shape[1]
        image_padded = cv2.copyMakeBorder(image,
                                          top=(self.input_size - h) // 2,
                                          bottom=self.input_size - (self.input_size - h) // 2 - h,
                                          left=(self.input_size - w) // 2,
                                          right=self.input_size - (self.input_size - w) // 2 - w,
                                          borderType=cv2.BORDER_CONSTANT)
        mask_padded = cv2.copyMakeBorder(mask,
                                         top=(self.input_size - h) // 2,
                                         bottom=self.input_size - (self.input_size - h) // 2 - h,
                                         left=(self.input_size - w) // 2,
                                         right=self.input_size - (self.input_size - w) // 2 - w,
                                         borderType=cv2.BORDER_CONSTANT)
        return image_padded, mask_padded

    @staticmethod
    def random_rotation(image: np.ndarray, mask: np.ndarray):
        """
        Random rotation image with degree
        :param image: np.ndarray
            Input image
        :param mask: np.ndarray
            Input mask
        :return: np.ndarray
            Image and mask after rotate.
        """
        degree = np.random.uniform(0, 360)
        h = image.shape[0]
        w = image.shape[1]
        rot_map = cv2.getRotationMatrix2D((w//2, h//2), degree, scale=1)
        image_rotated = cv2.warpAffine(image, rot_map, (w, h))
        mask_rotated = cv2.warpAffine(mask, rot_map, (w, h))
        return image_rotated, mask_rotated

    @staticmethod
    def random_flip(image: np.ndarray, mask: np.ndarray):
        """
        Random flip image
        :param image: np.ndarray
            Input image
        :param mask:
            Input mask
        :return: np.ndarray
            Image and mask after random flip
        """
        if np.random.choice([0, 1]):
            flip_code = np.random.choice([-1, 0, 1])
            image_flipped = cv2.flip(image, flipCode=flip_code)
            mask_flipped = cv2.flip(mask, flipCode=flip_code)
            return image_flipped, mask_flipped
        return image, mask

    @staticmethod
    def random_blur(image: np.ndarray, mask: np.ndarray):
        """
        Random blur image
        :param image: np.ndarray
            Input image
        :param mask: np.ndarray
            Input mask
        :return: np.ndarray
            Image and mask after random blur
        """
        if np.random.choice([0, 1]):
            radius = np.random.choice([1, 3, 5])
            image_blur = cv2.GaussianBlur(image, ksize=(radius, radius), sigmaX=1)
            mask_blur = cv2.GaussianBlur(mask, ksize=(radius, radius), sigmaX=1)
            return image_blur, mask_blur
        return image, mask

    @staticmethod
    def random_add_brightness(image: np.ndarray):
        """
        Random add brightness into image
        :param image: np.ndarray
            Input image
        :return: np.ndarray
            Image after add brightness
        """
        value = np.random.randint(0, 50)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        image_brightness = cv2.merge((h, s, v))
        image_brightness = cv2.cvtColor(image_brightness, cv2.COLOR_HSV2BGR)
        return image_brightness

    def data_process(self, image: np.ndarray, mask: np.ndarray):
        """
        Process all step for augmentation data.
        :param image: np.ndarray
            Input image
        :param mask: np.ndarray
            Input mask
        :return:
            Image and mask after process.
        """
        image_processed, mask_processed = self.resize_image(image, mask)
        image_processed, mask_processed = self.padding_image(image_processed, mask_processed)
        image_processed, mask_processed = self.random_rotation(image_processed, mask_processed)
        image_processed, mask_processed = self.random_flip(image_processed, mask_processed)
        image_processed, mask_processed = self.random_blur(image_processed, mask_processed)
        image_processed = self.random_add_brightness(image_processed)
        image_processed = cv2.cvtColor(image_processed, cv2.COLOR_RGB2GRAY)
        mask_processed = cv2.resize(mask_processed, dsize=(self.output_size, self.output_size))
        (thresh, mask_processed) = cv2.threshold(mask_processed, 127, 255, cv2.THRESH_BINARY)
        image_processed = image_processed/255.0
        mask_processed = mask_processed/255.0
        return image_processed, mask_processed

    def data_process_test(self, image: np.ndarray):
        """
        Process data for testing phase
        :param image: np.ndarray
            Input test image
        :param mask: np.ndarray
            Input test mask
        :return: np.ndarray
            Image and mask of test data after process
        """
        w = image.shape[1]
        h = image.shape[0]
        ratio = min(w, h) / max(w, h)
        if w >= h:
            new_w = self.input_size
            new_h = round(self.input_size * ratio)
        else:
            new_h = self.input_size
            new_w = round(self.input_size * ratio)
        image_resized = cv2.resize(image, (new_w, new_h))
        image_padded = cv2.copyMakeBorder(image_resized,
                                          top=(self.input_size - new_h) // 2,
                                          bottom=self.input_size - (self.input_size - new_h) // 2 - new_h,
                                          left=(self.input_size - new_w) // 2,
                                          right=self.input_size - (self.input_size - new_w) // 2 - new_w,
                                          borderType=cv2.BORDER_CONSTANT)
        image_processed = cv2.cvtColor(image_padded, cv2.COLOR_RGB2GRAY)
        image_processed = image_processed/255.0
        return image_processed
