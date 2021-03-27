import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, Conv2DTranspose
from tensorflow.keras.layers import Concatenate, BatchNormalization


class Unet:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = tf.keras.Sequential()
        self.L = 0
        self.inputs = None

    def down_sampling_block(self, inputs, num_conv_block, num_filter, kernel_size, activation):
        for i in range(num_conv_block):
            if self.L == 0:
                inputs = Conv2D(filters=num_filter, kernel_size=kernel_size,
                                activation=activation, padding='same')(self.inputs)
            else:
                inputs = Conv2D(filters=num_filter, kernel_size=kernel_size,
                                activation=activation, padding='same')(inputs)
            self.L += 1
        inputs = BatchNormalization()(inputs)
        outputs = MaxPool2D(pool_size=(2, 2))(inputs)
        self.L += 1
        return outputs, inputs

    def conv_block(self, inputs, num_conv_block, num_filter, kernel_size, activation):
        for i in range(num_conv_block):
            if self.L == 0:
                inputs = Conv2D(filters=num_filter, kernel_size=kernel_size,
                                activation=activation, padding='same')(self.inputs)
            else:
                inputs = (Conv2D(filters=num_filter, kernel_size=kernel_size,
                                 activation=activation, padding='same'))(inputs)
            self.L += 1
        outputs = inputs
        return outputs

    def up_sampling_block(self, inputs, concat, num_conv_block, num_filter, kernel_size, activation):
        for i in range(num_conv_block):
            if self.L == 0:
                inputs = Conv2D(filters=num_filter, kernel_size=kernel_size,
                                activation=activation, padding='same')(self.inputs)
            else:
                inputs = (Conv2D(filters=num_filter, kernel_size=kernel_size,
                                 activation=activation, padding='same'))(inputs)
            self.L += 1
        inputs = BatchNormalization()(inputs)
        outputs = Conv2DTranspose(filters=num_filter//2, kernel_size=kernel_size,
                                  strides=(2, 2), padding='same')(inputs)
        shape_enc = concat.shape[1]
        shape_dec = outputs.shape[1]
        index = (shape_enc-shape_dec)//2
        concat = concat[:, index:index+shape_dec, index:index+shape_dec, :]
        outputs = Concatenate()([concat, outputs])
        return outputs

    def build(self):
        self.inputs = Input(shape=self.input_shape)
        down1, concat1 = self.down_sampling_block(inputs=None, num_conv_block=2, num_filter=32,
                                                  kernel_size=(3, 3), activation='relu')
        down2, concat2 = self.down_sampling_block(inputs=down1, num_conv_block=2, num_filter=64,
                                                  kernel_size=(3, 3), activation='relu')
        down3, concat3 = self.down_sampling_block(inputs=down2, num_conv_block=2, num_filter=128,
                                                  kernel_size=(3, 3), activation='relu')
        down4, concat4 = self.down_sampling_block(inputs=down3, num_conv_block=2, num_filter=256,
                                                  kernel_size=(3, 3), activation='relu')
        up1 = self.up_sampling_block(inputs=down4, concat=concat4, num_conv_block=2, num_filter=512,
                                     kernel_size=(3, 3), activation='relu')
        up2 = self.up_sampling_block(inputs=up1, concat=concat3, num_conv_block=2, num_filter=256,
                                     kernel_size=(3, 3), activation='relu')
        up3 = self.up_sampling_block(inputs=up2, concat=concat2, num_conv_block=2, num_filter=128,
                                     kernel_size=(3, 3), activation='relu')
        up4 = self.up_sampling_block(inputs=up3, concat=concat1, num_conv_block=2, num_filter=64,
                                     kernel_size=(3, 3), activation='relu')
        conv1 = self.conv_block(inputs=up4, num_conv_block=2, num_filter=32,
                                kernel_size=(3, 3), activation='relu')
        output = self.conv_block(inputs=conv1, num_conv_block=1, num_filter=1,
                                 kernel_size=(1, 1), activation='sigmoid')
        model = Model(self.inputs, output)
        return model
