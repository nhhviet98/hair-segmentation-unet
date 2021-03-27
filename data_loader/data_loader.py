from data_augmentation.data_augmentation import DataAugmentation
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence


class DataLoader(Sequence):
    def __init__(self, meta_data_path: str, batch_size: int, abs_image_path: str, abs_mask_path: str,
                 phase: str = 'train', input_size: int = 224, output_size: int = 224):
        """
        Init method for Data Loader Class
        :param meta_data_path: str
            json path of dataset
        :param batch_size: int
            Batch size
        :param phase: str
            "train" or "test" phase
        :param input_size: int
            size of input image to fit with Unet input
        :param output_size:
            size of output mask to fit with Unet input
        """
        self.meta_data_path = meta_data_path
        self.batch_size = batch_size
        self.phase = phase
        self.input_size = input_size
        self.output_size = output_size
        self.abs_image_path = abs_image_path
        self.abs_mask_path = abs_mask_path
        self.train_path = None
        self.test_path = None
        self.indexes = None
        self.indexes_test = None
        self.read_meta_data()

    def read_meta_data(self):
        """
        Read meta data file
        :return: str
            file path of training set and testing set
        """
        files = open(self.meta_data_path)
        files = json.load(files)
        self.train_path = files['train'][:]
        self.test_path = files['test'][:]
        self.indexes = np.arange(len(self.train_path))
        self.indexes_test = np.arange(len(self.test_path))
        return self.train_path, self.test_path

    def process_image(self, image_paths: list):
        """
        Read and process image to fit with model
        :param image_paths: list
            List of image path of data set
        :return: list
            List of input and output tensor have size = batch size
        """
        data_transform = DataAugmentation(input_size=self.input_size, output_size=self.output_size)
        x_train = []
        y_train = []
        for path in image_paths:
            image = cv2.imread(self.abs_image_path + path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # original image in jpg format but mask in png format
            mask = cv2.imread(self.abs_mask_path + path[:len(path) - 3] + "png", cv2.IMREAD_GRAYSCALE)
            image_processed, mask_processed = data_transform.data_process(image, mask)
            image_processed = np.expand_dims(image_processed, axis=2)
            mask_processed = np.expand_dims(mask_processed, axis=2)
            x_train.append(tf.convert_to_tensor(image_processed))
            y_train.append(tf.convert_to_tensor(mask_processed))
        return x_train, y_train

    def __getitem__(self, index: int):
        """
        Get item method to get batch of data when use model.fit_generator
        :param index: int
            index of batch
        :return: tf.tensor
            tensor of image and mask for training phase
        """
        if self.phase == 'train':
            index_list = self.indexes
            data_path = self.train_path
        else:
            index_list = self.indexes_test
            data_path = self.test_path
        if index == self.__len__()-1:
            indexes = index_list[index * self.batch_size:]
        else:
            indexes = index_list[index*self.batch_size:(index+1)*self.batch_size]
        image_paths = [data_path[k] for k in indexes]
        image, mask = self.process_image(image_paths)
        image = tf.convert_to_tensor(image)
        mask = tf.convert_to_tensor(mask)
        return image, mask

    def __len__(self):
        """
        len method to calculate number of batch size
        :return: int
            number of batch size of data set
        """
        if self.phase == 'train':
            num_path = len(self.train_path)
        else:
            num_path = len(self.test_path)
        return int(np.ceil(num_path / self.batch_size))

