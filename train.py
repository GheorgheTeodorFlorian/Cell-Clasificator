from dataload import CellTypeDataLoader
import constants
from utils import display_examples, non_shuffling_train_test_split, plot_accuracy_loss_chart
from network import CNNModel
from interfaces import Trainer

import zope.interface
import random
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from keras.callbacks import ReduceLROnPlateau

import tensorflow as tf 

@zope.interface.implementer(Trainer)
class CNNTrainer():
    def __init__(self):
        random.seed(constants.randomSeed)
        self.dataLoader = CellTypeDataLoader()

    def prepareData(self):
        images, labels, imageNames = self.dataLoader.load_data()
        images, labels, imageNames = shuffle(images, labels, imageNames, random_state=10)

        train_images, test_images, train_labels, test_labels = non_shuffling_train_test_split(images, labels, test_size = 0.2)
        test_images, val_images, test_labels, val_labels = non_shuffling_train_test_split(test_images, test_labels, test_size = 0.5)

        _, train_counts = np.unique(train_labels, return_counts = True)
        _, val_counts = np.unique(val_labels, return_counts = True)
        _, test_counts = np.unique(test_labels, return_counts = True)

        pd.DataFrame({'train': train_counts, "val": val_counts, "test": test_counts}, index = constants.class_names).plot.bar()

        plt.show()

        plt.pie(train_counts,
                explode=(0, 0, 0, 0) , 
                labels=constants.class_names,
                autopct='%1.1f%%')
        plt.axis('equal')
        plt.title('Proportion of each observed category')
        plt.show()

        train_images = train_images / 255.0 
        val_images = val_images / 255.0
        test_images = test_images / 255.0

        display_examples(constants.class_names, train_images, train_labels)

        return train_images, val_images, test_images, train_labels, val_labels, test_labels

    def train(self):
        tf.compat.v1.debugging.set_log_device_placement(True)
        train_images, val_images, test_images, train_labels, val_labels, test_labels = self.prepareData()
       
        learning_rate_reduction = ReduceLROnPlateau(
            monitor = 'val_accuracy', 
            patience = 2, 
            verbose = 1, 
            factor = 0.3, 
            min_lr = 0.000001)

        model = CNNModel().model()
        # Train
        history = model.fit(
            train_images, 
            train_labels, 
            batch_size = 32, 
            epochs = 30, 
            validation_data=(val_images, val_labels), 
            callbacks=[learning_rate_reduction ])

        plot_accuracy_loss_chart(history)


        results = model.evaluate(test_images, test_labels)

        print("Loss of the model  is - test ", results[0])
        print("Accuracy of the model is - test", results[1]*100, "%")


        results = model.evaluate(val_images, val_labels)

        print("Loss of the model  is - val ", results[0])
        print("Accuracy of the model is - val", results[1]*100, "%")

        results = model.evaluate(train_images, train_labels)

        print("Loss of the model  is - train ", results[0])
        print("Accuracy of the model is - train", results[1]*100, "%")

        model.save('model.h5')

trainer = CNNTrainer()
trainer.train()