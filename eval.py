import zope.interface
from interfaces import Evaluate

from dataload import CellTypeDataLoader
import constants
from utils import display_examples, non_shuffling_train_test_split, plot_confusion_matrix

import random
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix

@zope.interface.implementer(Evaluate)
class EvaluateCellTypeClassfier:
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

    def evaluate(self):
        train_images, val_images, test_images, train_labels, val_labels, test_labels = self.prepareData()
        
        model = load_model("model.h5")

        predictions = model.predict(test_images)
        predictions = np.argmax(predictions,axis=1)
        print(classification_report(
            test_labels, 
            predictions, 
            target_names = ['EOSINOPHIL (Class 0)', 'LYMPHOCYTE (Class 1)', 'MONOCYTE (Class 2)', 'NEUTROPHIL (Class 3)']))
        print(accuracy_score(test_labels, predictions))
        cm = confusion_matrix(test_labels, predictions)
        cm = pd.DataFrame(cm, index = ['0', '1', '2', '3'], columns = ['0', '1', '2', '3'])
        plot_confusion_matrix(cm)

eval = EvaluateCellTypeClassfier()
eval.evaluate()
