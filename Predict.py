import tensorflow as tf
import cv2
from models.unet import Unet
from data_augmentation.data_augmentation import DataAugmentation
import numpy as np


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Initialize
IMAGE_PATH = "dataset/Original/Testing/"
MASK_PATH = "dataset/MASKS/Testing/"
IMAGE_FILE = "Frame00314-org"

model = Unet(input_shape=(224, 224, 1)).build()
model.load_weights("models/model_weight.h5")
model.summary()
print("yeah")



def convert_to_tensor(numpy_image):
    numpy_image = np.expand_dims(numpy_image, axis=2)
    numpy_image = np.expand_dims(numpy_image, axis=0)
    tensor_image = tf.convert_to_tensor(numpy_image)
    return tensor_image


def predict(image):
    process_obj = DataAugmentation(input_size=224, output_size=224)
    image_processed = process_obj.data_process_test(image)
    tensor_image = convert_to_tensor(image_processed)
    predicted_mask = model.predict(tensor_image)
    predicted_mask = np.array(predicted_mask).squeeze()
    predicted_mask = np.where(predicted_mask > 0.5, 1, 0)
    return predicted_mask

if __name__ == '__main__':
    img = cv2.imread(IMAGE_PATH + IMAGE_FILE + ".jpg")
    mask = predict(img)
    print("End program")
