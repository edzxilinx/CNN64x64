from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Dropout, BatchNormalization
from keras.regularizers import l2


reg = 0.0001  # @param
bnEps = 2e-5  # @param
bnMom = 0.9  # @param
BS_logic = True  # @param
input_shape = (64, 64, 3)  # @param
chan_dim = -1  # @param
kernel_size = (3, 3)  # @param
pool_size = (2, 2)  # @param
filters_base = 32  # @param


class CnnCreator:

    def __init__(self):
        pass

    @staticmethod
    def wastes_v1_arch(input_shape, ksize, psize, filter_base):
        net = Sequential()

        net.add(Conv2D(filters=filters_base, kernel_size=kernel_size, padding="same", use_bias=BS_logic,
                       kernel_regularizer=l2(reg), input_shape=input_shape))
        net.add(BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom))
        net.add(Activation(activation='relu'))

        net.add(Conv2D(filters=filters_base, kernel_size=kernel_size, padding="same", use_bias=BS_logic,
                       kernel_regularizer=l2(reg)))

        net.add(BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom))
        net.add(Activation(activation='relu'))

        net.add(MaxPooling2D(pool_size=pool_size))
        net.add(Dropout(rate=0.3))

        net.add(Conv2D(filters=filters_base * 2, kernel_size=kernel_size, padding="same", use_bias=BS_logic,
                       kernel_regularizer=l2(reg)))
        net.add(BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom))
        net.add(Activation(activation='relu'))

        net.add(Conv2D(filters=filters_base * 2, kernel_size=kernel_size, padding="same", use_bias=BS_logic,
                       kernel_regularizer=l2(reg)))
        net.add(BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom))
        net.add(Activation(activation='relu'))

        net.add(MaxPooling2D(pool_size=pool_size))
        net.add(Dropout(rate=0.3))

        net.add(Conv2D(filters=filters_base * 4, kernel_size=kernel_size, padding="same", use_bias=BS_logic,
                       kernel_regularizer=l2(reg)))
        net.add(BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom))
        net.add(Activation(activation='relu'))

        net.add(Conv2D(filters=filters_base * 4, kernel_size=kernel_size, padding="same", use_bias=BS_logic,
                       kernel_regularizer=l2(reg)))
        net.add(BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom))
        net.add(Activation(activation='relu'))

        net.add(MaxPooling2D(pool_size=pool_size))
        net.add(Dropout(rate=0.3))

        net.add(Conv2D(filters=filters_base * 8, kernel_size=kernel_size, padding="same", use_bias=BS_logic,
                       kernel_regularizer=l2(reg)))
        net.add(BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom))
        net.add(Activation(activation='relu'))

        net.add(Conv2D(filters=filters_base * 8, kernel_size=kernel_size, padding="same", use_bias=BS_logic,
                       kernel_regularizer=l2(reg)))
        net.add(BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom))
        net.add(Activation(activation='relu'))

        net.add(MaxPooling2D(pool_size=pool_size))
        net.add(Dropout(rate=0.3))

        net.add(Conv2D(filters=filters_base * 16, kernel_size=kernel_size, padding="same", use_bias=BS_logic,
                       kernel_regularizer=l2(reg)))
        net.add(BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom))
        net.add(Activation(activation='relu'))

        net.add(Conv2D(filters=filters_base * 16, kernel_size=kernel_size, padding="same", use_bias=BS_logic,
                       kernel_regularizer=l2(reg)))
        net.add(BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom))
        net.add(Activation(activation='relu'))

        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(rate=0.3))

        net.add(Conv2D(filters=filters_base * 32, kernel_size=kernel_size, padding="same", use_bias=BS_logic,
                       kernel_regularizer=l2(reg)))
        net.add(BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom))
        net.add(Activation(activation='relu'))

        net.add(Conv2D(filters=filters_base * 32, kernel_size=kernel_size, padding="same", use_bias=BS_logic,
                       kernel_regularizer=l2(reg)))
        net.add(BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom))
        net.add(Activation(activation='relu'))

        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Dropout(rate=0.3))

        net.add(Flatten())
        net.add(Dense(units=1024))
        net.add(BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom))
        net.add(Activation('relu'))
        net.add(Dropout(rate=0.7))
        net.add(Dense(units=3, kernel_regularizer=l2(reg)))
        net.add(Activation(activation='softmax'))

        return net

