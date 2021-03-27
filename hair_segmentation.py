import tensorflow as tf
from data_loader.data_loader import DataLoader
from models.unet import Unet


def scheduler(epoch, lr):
    if epoch != 0 and epoch % 5 == 0:
        return lr * 0.3
    else:
        return lr


def iou(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # Init all path of this project
    META_DATA_PATH = "dataset/data.json"
    TRAINING_DATA_PATH = "dataset/Original/Training/"
    TRAINING_MASK_PATH = "dataset/MASKS/Training/"
    TESTING_DATA_PATH = "dataset/Original/Testing/"
    TESTING_MASK_PATH = "dataset/MASKS/Testing/"

    # Hyper parameters
    smooth = 1e-15
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 10
    MOMENTUM = 0.9
    NUM_CLASS = 1

    # Data Loader
    train_generator = DataLoader(META_DATA_PATH, batch_size=BATCH_SIZE,
                                 abs_image_path=TRAINING_DATA_PATH, abs_mask_path=TRAINING_MASK_PATH,
                                 phase='train', input_size=224, output_size=224)
    test_generator = DataLoader(META_DATA_PATH, batch_size=BATCH_SIZE,
                                abs_image_path=TESTING_DATA_PATH, abs_mask_path=TESTING_MASK_PATH,
                                phase='test', input_size=224, output_size=224)

    # Build model using Unet class
    model = Unet(input_shape=(224, 224, 1)).build()
    losses = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(optimizer=optimizer, loss=losses, metrics=iou)

    # Training model with my custom generator
    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=EPOCHS,
                        callbacks=[callback],
                        validation_data=test_generator,
                        validation_steps=len(test_generator))
