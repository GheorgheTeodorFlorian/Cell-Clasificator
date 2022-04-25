
import zope.interface
from interfaces import Network
from keras.models import Sequential
from keras.layers import SeparableConv2D, BatchNormalization, Dropout, MaxPooling2D, Input, Dense, Conv2D, Activation, Flatten 

@zope.interface.implementer(Network)
class CNNModel:
    def model(self):
        model = Sequential()

        # First Conv block
        model.add(Conv2D(16 , (3,3) , padding = 'same' , activation = 'relu' , input_shape = (120,120,3)))
        model.add(Conv2D(16 , (3,3), padding = 'same' , activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))

        # Second Conv block
        model.add(SeparableConv2D(32, (3,3), activation = 'relu', padding = 'same'))
        model.add(SeparableConv2D(32, (3,3), activation = 'relu', padding = 'same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2,2)))

        # Third Conv block
        model.add(SeparableConv2D(64, (3,3), activation = 'relu', padding = 'same'))
        model.add(SeparableConv2D(64, (3,3), activation = 'relu', padding = 'same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2,2)))

        # Forth Conv block
        model.add(SeparableConv2D(128, (3,3), activation = 'relu', padding = 'same'))
        model.add(SeparableConv2D(128, (3,3), activation = 'relu', padding = 'same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(0.2))

        # Fifth Conv block 
        model.add(SeparableConv2D(256, (3,3), activation = 'relu', padding = 'same'))
        model.add(SeparableConv2D(256, (3,3), activation = 'relu', padding = 'same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(0.2))


        # FC layer 
        model.add(Flatten())
        model.add(Dense(units = 512 , activation = 'tanh'))
        model.add(Dropout(0.7))
        model.add(Dense(units = 128 , activation = 'tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(units = 64 , activation = 'tanh'))
        model.add(Dropout(0.3))

        # Output layer
        model.add(Dense(units = 4 , activation = 'softmax'))

        # Compile
        model.compile(optimizer = "adam" , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])
        model.summary()

        return model

