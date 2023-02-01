import h5py
import glob
from numpy.random import seed
import tensorflow as tf


from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model

from VQA.quantum_Circuit_Sample import VQA


class DeepLearningModel():

    """
    Implement the model
    """
    def predict_result(self, current_image, current_car_state):
        result = self.model.predict([current_image, current_car_state])
        return result