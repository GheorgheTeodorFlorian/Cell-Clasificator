import zope.interface

class DataLoaderInterface(zope.interface.Interface):
    def load_data(self):
        pass

class Network(zope.interface.Interface):
    def model(self):
        pass

class Trainer(zope.interface.Interface):
    def train(self):
        pass

class Evaluate(zope.interface.Interface):
    def evaluate(self):
        pass