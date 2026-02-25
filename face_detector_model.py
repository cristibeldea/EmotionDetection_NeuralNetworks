from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization


def build_face_detector(input_shape=(64, 64, 3)):  # changed from 32x32

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(64, (1, 1)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(128, (3, 3)))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (1, 1)))
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.summary()
    return model

    # model = Sequential()
    #
    # model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(64, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Flatten())
    # model.add(Dense(64))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    #
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #return model

    # model = Sequential()
    #
    # # Block 1
    # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # # Block 2
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # # Block 3
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    #
    # # Block 4
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # # Dense head
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))  # For face detection
    #
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #return model
