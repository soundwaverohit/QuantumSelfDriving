import h5py
import glob
from numpy.random import seed
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, Input, concatenate, MaxPooling2D
from tensorflow.keras.models import Model, Sequential

from ml.additional.generator import DriveDataGenerator
from ml.additional.utils import *

from Quantum_Circuit import MyVQAClass


class DeepLearningModel():
    TRAINING_H5_FILE_NAME = "train.h5"
    VALIDATION_H5_FILE_NAME = "val.h5"
    H5_PROPERTY_LABEL = "label"
    H5_PROPERTY_PREVIOUS_STATE = "previous_state"
    H5_PROPERTY_IMAGE = "image"
    MODEL_LOG_FILE = "training_log.csv"
    MODEL_NAME_FORMAT = "model.{0}-{1}.h5"
    MODEL_NAME_FORMAT_FIRST_PART = "{epoch:02d}",
    MODEL_NAME_FORMAT_SECOND_PART = "{val_loss:.7f}"
    MODEL_LOADING_REGEX = "*.h5"
    processed_folder = None
    model_output_folder = None
    train_dataset = None
    val_dataset = None
    data_generator = None
    train_generator = None
    val_generator = None
    callbacks = None
    db_connection = None

    image_input_shape = None
    real_image_shape_with_roi = (1, 59, 255, 3)
    state_input_shape = None
    model = None
    history = None

    # Default values
    values = DictAttr(
        seed_number=1,
        batch_size=32,
        train_zero_drop_percentage=0.3,
        val_zero_drop_percentage=0.3,
        # Region of interest
        roi=[76, 135, 0, 255],
        activation_function="relu",
        padding="same",
        # Data Generator params
        data_generator_rescale=1. / 255.,
        data_generator_horizontal_flip=True,
        data_generator_brighten_range=0.4,
        # CNN params
        layer_0_kernel_size=16,
        layer_0_strides=(3, 3),
        layer_0_pooling_size=(2, 2),
        layer_1_kernel_size=32,
        layer_1_strides=(3, 3),
        layer_1_pooling_size=(2, 2),
        layer_2_kernel_size=32,
        layer_2_strides=(3, 3),
        layer_2_pooling_size=(2, 2),
        cnn_dropout=0.2,
        dense_0_units=64,
        dense_0_dropout=0.2,
        dense_1_units=10,
        dense_1_dropout=0.2,
        dense_2_units=1,
        # Learning params
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        # minimum square error
        loss_function="mse",
        monitored_quantity="val_loss",
        # plateau params
        # factor: factor by which the learning rate will be reduced. new_lr = lr * factor
        # patience: number of epochs with no improvement after which learning rate will be reduced.
        plateau_factor=0.5,
        plateau_patience=3,
        plateau_min_learning_rate=0.0001,
        plateau_verbose=1,
        # early_stopping params
        # min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
        #   change of less than min_delta, will count as no improvement.
        # patience: number of epochs with no improvement after which training will be stopped.
        early_stopping_patience=30,
        early_stopping_min_delta=0,
        early_stopping_verbose=1,
        # Model Checkpoint params
        checkpoint_save_best_only=True,
        checkpoint_verbose=1,
        # Model training
        model_training_epoche=200,
        model_training_verbose=2,
        trainable=False,
        hybrid=True,
        qml_trainable=True
    )
    vqa_obj = None
    hybrid_type = 0

    def __init__(self, processed_folder, model_output_folder, batch_size=32, trained_rotations=None, hybrid_type=0):
        self.processed_folder = processed_folder
        self.model_output_folder = model_output_folder
        self.batch_size = batch_size
        self.vqa_obj = MyVQAClass(trained_rotations)
        self.hybrid_type = hybrid_type

    def change_default_params(self, new_model_params):
        for key in new_model_params:
            if key in self.values:
                self.values[key] = new_model_params[key]
        print("Default params changed!")

    def _load_model(self, path,):
        self.load_data()
        self.init_data_generator()
        self.create_model()
        self.model.load_weights(path, by_name=True)
        #maybe for explainability
        #layer_outputs = [layer.output for layer in model.layers[:16]]  # Extracts the outputs of the top layers
        #self.model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

    def load_model(self, model_name):
        self._load_model(model_name)

    def get_latest_model_path(self):
        list_of_models = glob.glob(os.path.join(self.model_output_folder, self.MODEL_LOADING_REGEX))
        if len(list_of_models) > 0:
            latest_model = max(list_of_models, key=os.path.getctime)
            return latest_model
        return None

    def get_image_buf_shape_and_roi(self):
        return [self.real_image_shape_with_roi, self.values.roi]

    def reset_model(self):
        self.model = None

    def start_training(self, epoch=None):
        self.load_data()
        self.init_data_generator()
        self.create_model()
        self.create_callback_functions()
        self.start_with_model_fit(epoch)

    def load_data(self):
        self.train_dataset = h5py.File(os.path.join(self.processed_folder, self.TRAINING_H5_FILE_NAME), 'r')
        self.val_dataset = h5py.File(os.path.join(self.processed_folder, self.VALIDATION_H5_FILE_NAME), 'r')

    def init_data_generator(self):
        self.data_generator = DriveDataGenerator(rescale=self.values.data_generator_rescale,
                                                 horizontal_flip=self.values.data_generator_horizontal_flip,
                                                 brighten_range=self.values.data_generator_brighten_range)
        self.train_generator = self.data_generator.flow(self.train_dataset[self.H5_PROPERTY_IMAGE],
                                                        self.train_dataset[self.H5_PROPERTY_PREVIOUS_STATE],
                                                        self.train_dataset[self.H5_PROPERTY_LABEL],
                                                        batch_size=self.batch_size,
                                                        zero_drop_percentage=self.values.train_zero_drop_percentage,
                                                        roi=self.values.roi)
        self.val_generator = self.data_generator.flow(self.val_dataset[self.H5_PROPERTY_IMAGE],
                                                      self.val_dataset[self.H5_PROPERTY_PREVIOUS_STATE],
                                                      self.val_dataset[self.H5_PROPERTY_LABEL],
                                                      batch_size=self.batch_size,
                                                      zero_drop_percentage=self.values.val_zero_drop_percentage,
                                                      roi=self.values.roi)

        [sample_batch_train_data, sample_batch_y_data] = next(self.train_generator)
        next(self.val_generator)

        # Define input shape
        self.image_input_shape = sample_batch_train_data[0].shape[1:]
        self.state_input_shape = sample_batch_train_data[1].shape[1:]

    def create_model(self):
        seed(self.values.seed_number)
        #tf.set_random_seed(self.values.seed_number)

        trainable = self.values.trainable
        hybrid = self.values.hybrid
        qml_trainable = self.values.qml_trainable

        # Create the convolutional stacks
        pic_input = Input(shape=self.image_input_shape)

        img_stack = Conv2D(self.values.layer_0_kernel_size, self.values.layer_0_strides,
                           activation=self.values.activation_function,
                           padding=self.values.padding, name="convolution0", trainable=trainable)(pic_input)
        img_stack = MaxPooling2D(pool_size=self.values.layer_0_pooling_size, trainable=trainable)(img_stack)
        img_stack = Conv2D(self.values.layer_1_kernel_size, self.values.layer_1_strides,
                           activation=self.values.activation_function,
                           padding=self.values.padding, name='convolution1', trainable=trainable)(img_stack)
        img_stack = MaxPooling2D(pool_size=self.values.layer_1_pooling_size, trainable=trainable)(img_stack)
        img_stack = Conv2D(self.values.layer_2_kernel_size, self.values.layer_2_strides,
                           activation=self.values.activation_function,
                           padding=self.values.padding, name='convolution2', trainable=trainable)(img_stack)
        img_stack = MaxPooling2D(pool_size=self.values.layer_2_pooling_size, trainable=trainable)(img_stack)
        img_stack = Flatten()(img_stack)
        img_stack = Dropout(self.values.cnn_dropout, trainable=trainable)(img_stack)

        # Inject the state input
        state_input = Input(shape=self.state_input_shape)
        merged = tf.keras.layers.Concatenate()([img_stack, state_input])

        output_dense = None
        if hybrid:
            # Add a few dense layers to finish the model
            merged = Dense(self.values.dense_0_units, activation=self.values.activation_function, name='dense_qnn0')(
                merged)
            merged = Dropout(self.values.dense_0_dropout)(merged)
            # different hybrid VQA circuits
            if self.hybrid_type == 0:
                merged = Dense(4, activation=self.values.activation_function, name='dense_pre_out')(merged)
                merged = Dropout(self.values.dense_1_dropout)(merged)
                #VQA layer
                # 4 qubits, 4 input and 4 output
                qlayer = self.vqa_obj.create_two_layer_circuit(trainable=qml_trainable)
                x = qlayer(merged)
                x = Dense(4, activation=self.values.activation_function, name='dense_qnn2')(x)
                output_dense = Dense(self.values.dense_2_units, name='output_qnn')(x)
            elif self.hybrid_type == 1:
                merged = Dense(4, activation=self.values.activation_function, name='dense_pre_out')(merged)
                merged = Dropout(self.values.dense_1_dropout)(merged)
                # VQA layer
                # 6 qubits, 4 input and 2 output
                qlayer = self.vqa_obj.create_custom_v1_circuit(trainable=qml_trainable)
                x = qlayer(merged)
                x = Dense(2, activation=self.values.activation_function, name='dense_qnn2')(x)
                output_dense = Dense(self.values.dense_2_units, name='output_qnn')(x)
            elif self.hybrid_type == 2:
                merged = Dense(4, activation=self.values.activation_function, name='dense_pre_out')(merged)
                merged = Dropout(self.values.dense_1_dropout)(merged)
                # VQA layer
                # 6 qubits, 4 input and 2 output - additional gates compared to hybrid_type 1
                qlayer = self.vqa_obj.create_custom_v2_circuit(trainable=qml_trainable)
                x = qlayer(merged)
                x = Dense(2, activation=self.values.activation_function, name='dense_qnn2')(x)
                output_dense = Dense(self.values.dense_2_units, name='output_qnn')(x)
            elif self.hybrid_type == 3:
                merged = Dense(4, activation=self.values.activation_function, name='dense_pre_out')(merged)
                merged = Dropout(self.values.dense_1_dropout)(merged)
                # VQA layer
                # 4 qubits, 4 input and 2 output
                qlayer = self.vqa_obj.create_custom_v3_circuit(trainable=qml_trainable)
                x = qlayer(merged)
                x = Dense(2, activation=self.values.activation_function, name='dense_qnn2')(x)
                output_dense = Dense(self.values.dense_2_units, name='output_qnn')(x)
            elif self.hybrid_type == 4:
                merged = Dense(4, activation=self.values.activation_function, name='dense_pre_out')(merged)
                merged = Dropout(self.values.dense_1_dropout)(merged)
                # VQA layer
                # 4 qubits, 4 input and 2 output - additional gates compared to hybrid_type 3
                qlayer = self.vqa_obj.create_custom_v4_circuit(trainable=qml_trainable)
                x = qlayer(merged)
                x = Dense(2, activation=self.values.activation_function, name='dense_qnn2')(x)
                output_dense = Dense(self.values.dense_2_units, name='output_qnn')(x)
            elif self.hybrid_type == 5:
                merged = Dense(4, activation=self.values.activation_function, name='dense_pre_out')(merged)
                merged = Dropout(self.values.dense_1_dropout)(merged)
                # VQA layer
                qlayer = self.vqa_obj.create_custom_v5_circuit(trainable=qml_trainable)
                x = qlayer(merged)
                x = Dense(2, activation=self.values.activation_function, name='dense_qnn2')(x)
                output_dense = Dense(self.values.dense_2_units, name='output_qnn')(x)
            elif self.hybrid_type == 6:
                merged = Dense(8, activation=self.values.activation_function, name='dense_pre_out')(merged)
                merged = Dropout(self.values.dense_1_dropout)(merged)
                # VQA layer
                qlayer = self.vqa_obj.create_custom_v6_circuit(trainable=qml_trainable)
                x = qlayer(merged)
                x = Dense(2, activation=self.values.activation_function, name='dense_qnn2')(x)
                output_dense = Dense(self.values.dense_2_units, name='output_qnn')(x)


        else:
            # Add a few dense layers to finish the model
            print("classic model architecture is used")
            merged = Dense(self.values.dense_0_units, activation=self.values.activation_function, name='dense0')(merged)
            merged = Dropout(self.values.dense_0_dropout)(merged)
            merged = Dense(self.values.dense_1_units, activation=self.values.activation_function, name='dense2')(merged)
            merged = Dropout(self.values.dense_1_dropout)(merged)
            output_dense = Dense(self.values.dense_2_units, name='output')(merged)

        self.model = Model(inputs=[pic_input, state_input], outputs=output_dense)
        adam = tf.keras.optimizers.Nadam(lr=self.values.learning_rate, beta_1=self.values.beta_1, beta_2=self.values.beta_2,
                     epsilon=self.values.epsilon)
        self.model.compile(optimizer=adam, loss=self.values.loss_function)

    def create_callback_functions(self):
        plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor=self.values.monitored_quantity, factor=self.values.plateau_factor,
                                             patience=self.values.plateau_patience,
                                             min_lr=self.values.plateau_min_learning_rate,
                                             verbose=self.values.plateau_verbose)
        checkpoint_filepath = os.path.join(self.model_output_folder,
                                           self.MODEL_NAME_FORMAT.format(self.MODEL_NAME_FORMAT_FIRST_PART,
                                                                         self.MODEL_NAME_FORMAT_SECOND_PART))
        check_and_create_dir(checkpoint_filepath)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_best_only=self.values.checkpoint_save_best_only,
                                              verbose=self.values.checkpoint_verbose)
        csv_callback = tf.keras.callbacks.CSVLogger(os.path.join(self.model_output_folder, self.MODEL_LOG_FILE))
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor=self.values.monitored_quantity,
                                                patience=self.values.early_stopping_patience,
                                                min_delta=self.values.early_stopping_min_delta,
                                                verbose=self.values.early_stopping_verbose)

        self.callbacks = [checkpoint_callback]#[plateau_callback, csv_callback, checkpoint_callback, early_stopping_callback]#,
                          #SaveInformationCallback(self.db_connection)]

    def start_with_model_fit(self, epoch=None):
        num_train_examples = self.train_dataset[self.H5_PROPERTY_IMAGE].shape[0]
        num_val_examples = self.val_dataset[self.H5_PROPERTY_IMAGE].shape[0]

        tmp_epoch = epoch
        if not tmp_epoch:
            tmp_epoch = self.values.model_training_epoche
        self.history = self.model.fit_generator(self.train_generator,
                                                steps_per_epoch=num_train_examples // self.batch_size,
                                                epochs=tmp_epoch,
                                                callbacks=self.callbacks, validation_data=self.val_generator,
                                                validation_steps=num_val_examples // self.batch_size,
                                                verbose=self.values.model_training_verbose)

    def predict_result(self, current_image, current_car_state):
        result = self.model.predict([current_image, current_car_state])
        return resultimport h5py
import glob
from numpy.random import seed
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, Input, concatenate, MaxPooling2D
from tensorflow.keras.models import Model, Sequential

from ml.additional.generator import DriveDataGenerator
from ml.additional.utils import *

from ml.vqa_models import MyVQAClass


class DeepLearningModel():
    TRAINING_H5_FILE_NAME = "train.h5"
    VALIDATION_H5_FILE_NAME = "val.h5"
    H5_PROPERTY_LABEL = "label"
    H5_PROPERTY_PREVIOUS_STATE = "previous_state"
    H5_PROPERTY_IMAGE = "image"
    MODEL_LOG_FILE = "training_log.csv"
    MODEL_NAME_FORMAT = "model.{0}-{1}.h5"
    MODEL_NAME_FORMAT_FIRST_PART = "{epoch:02d}",
    MODEL_NAME_FORMAT_SECOND_PART = "{val_loss:.7f}"
    MODEL_LOADING_REGEX = "*.h5"
    processed_folder = None
    model_output_folder = None
    train_dataset = None
    val_dataset = None
    data_generator = None
    train_generator = None
    val_generator = None
    callbacks = None
    db_connection = None

    image_input_shape = None
    real_image_shape_with_roi = (1, 59, 255, 3)
    state_input_shape = None
    model = None
    history = None

    # Default values
    values = DictAttr(
        seed_number=1,
        batch_size=32,
        train_zero_drop_percentage=0.3,
        val_zero_drop_percentage=0.3,
        # Region of interest
        roi=[76, 135, 0, 255],
        activation_function="relu",
        padding="same",
        # Data Generator params
        data_generator_rescale=1. / 255.,
        data_generator_horizontal_flip=True,
        data_generator_brighten_range=0.4,
        # CNN params
        layer_0_kernel_size=16,
        layer_0_strides=(3, 3),
        layer_0_pooling_size=(2, 2),
        layer_1_kernel_size=32,
        layer_1_strides=(3, 3),
        layer_1_pooling_size=(2, 2),
        layer_2_kernel_size=32,
        layer_2_strides=(3, 3),
        layer_2_pooling_size=(2, 2),
        cnn_dropout=0.2,
        dense_0_units=64,
        dense_0_dropout=0.2,
        dense_1_units=10,
        dense_1_dropout=0.2,
        dense_2_units=1,
        # Learning params
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        # minimum square error
        loss_function="mse",
        monitored_quantity="val_loss",
        # plateau params
        # factor: factor by which the learning rate will be reduced. new_lr = lr * factor
        # patience: number of epochs with no improvement after which learning rate will be reduced.
        plateau_factor=0.5,
        plateau_patience=3,
        plateau_min_learning_rate=0.0001,
        plateau_verbose=1,
        # early_stopping params
        # min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
        #   change of less than min_delta, will count as no improvement.
        # patience: number of epochs with no improvement after which training will be stopped.
        early_stopping_patience=30,
        early_stopping_min_delta=0,
        early_stopping_verbose=1,
        # Model Checkpoint params
        checkpoint_save_best_only=True,
        checkpoint_verbose=1,
        # Model training
        model_training_epoche=200,
        model_training_verbose=2,
        trainable=False,
        hybrid=True,
        qml_trainable=True
    )
    vqa_obj = None
    hybrid_type = 0

    def __init__(self, processed_folder, model_output_folder, batch_size=32, trained_rotations=None, hybrid_type=0):
        self.processed_folder = processed_folder
        self.model_output_folder = model_output_folder
        self.batch_size = batch_size
        self.vqa_obj = MyVQAClass(trained_rotations)
        self.hybrid_type = hybrid_type

    def change_default_params(self, new_model_params):
        for key in new_model_params:
            if key in self.values:
                self.values[key] = new_model_params[key]
        print("Default params changed!")

    def _load_model(self, path,):
        self.load_data()
        self.init_data_generator()
        self.create_model()
        self.model.load_weights(path, by_name=True)
        #maybe for explainability
        #layer_outputs = [layer.output for layer in model.layers[:16]]  # Extracts the outputs of the top layers
        #self.model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

    def load_model(self, model_name):
        self._load_model(model_name)

    def get_latest_model_path(self):
        list_of_models = glob.glob(os.path.join(self.model_output_folder, self.MODEL_LOADING_REGEX))
        if len(list_of_models) > 0:
            latest_model = max(list_of_models, key=os.path.getctime)
            return latest_model
        return None

    def get_image_buf_shape_and_roi(self):
        return [self.real_image_shape_with_roi, self.values.roi]

    def reset_model(self):
        self.model = None

    def start_training(self, epoch=None):
        self.load_data()
        self.init_data_generator()
        self.create_model()
        self.create_callback_functions()
        self.start_with_model_fit(epoch)

    def load_data(self):
        self.train_dataset = h5py.File(os.path.join(self.processed_folder, self.TRAINING_H5_FILE_NAME), 'r')
        self.val_dataset = h5py.File(os.path.join(self.processed_folder, self.VALIDATION_H5_FILE_NAME), 'r')

    def init_data_generator(self):
        self.data_generator = DriveDataGenerator(rescale=self.values.data_generator_rescale,
                                                 horizontal_flip=self.values.data_generator_horizontal_flip,
                                                 brighten_range=self.values.data_generator_brighten_range)
        self.train_generator = self.data_generator.flow(self.train_dataset[self.H5_PROPERTY_IMAGE],
                                                        self.train_dataset[self.H5_PROPERTY_PREVIOUS_STATE],
                                                        self.train_dataset[self.H5_PROPERTY_LABEL],
                                                        batch_size=self.batch_size,
                                                        zero_drop_percentage=self.values.train_zero_drop_percentage,
                                                        roi=self.values.roi)
        self.val_generator = self.data_generator.flow(self.val_dataset[self.H5_PROPERTY_IMAGE],
                                                      self.val_dataset[self.H5_PROPERTY_PREVIOUS_STATE],
                                                      self.val_dataset[self.H5_PROPERTY_LABEL],
                                                      batch_size=self.batch_size,
                                                      zero_drop_percentage=self.values.val_zero_drop_percentage,
                                                      roi=self.values.roi)

        [sample_batch_train_data, sample_batch_y_data] = next(self.train_generator)
        next(self.val_generator)

        # Define input shape
        self.image_input_shape = sample_batch_train_data[0].shape[1:]
        self.state_input_shape = sample_batch_train_data[1].shape[1:]

    def create_model(self):
        seed(self.values.seed_number)
        #tf.set_random_seed(self.values.seed_number)

        trainable = self.values.trainable
        hybrid = self.values.hybrid
        qml_trainable = self.values.qml_trainable

        # Create the convolutional stacks
        pic_input = Input(shape=self.image_input_shape)

        img_stack = Conv2D(self.values.layer_0_kernel_size, self.values.layer_0_strides,
                           activation=self.values.activation_function,
                           padding=self.values.padding, name="convolution0", trainable=trainable)(pic_input)
        img_stack = MaxPooling2D(pool_size=self.values.layer_0_pooling_size, trainable=trainable)(img_stack)
        img_stack = Conv2D(self.values.layer_1_kernel_size, self.values.layer_1_strides,
                           activation=self.values.activation_function,
                           padding=self.values.padding, name='convolution1', trainable=trainable)(img_stack)
        img_stack = MaxPooling2D(pool_size=self.values.layer_1_pooling_size, trainable=trainable)(img_stack)
        img_stack = Conv2D(self.values.layer_2_kernel_size, self.values.layer_2_strides,
                           activation=self.values.activation_function,
                           padding=self.values.padding, name='convolution2', trainable=trainable)(img_stack)
        img_stack = MaxPooling2D(pool_size=self.values.layer_2_pooling_size, trainable=trainable)(img_stack)
        img_stack = Flatten()(img_stack)
        img_stack = Dropout(self.values.cnn_dropout, trainable=trainable)(img_stack)

        # Inject the state input
        state_input = Input(shape=self.state_input_shape)
        merged = tf.keras.layers.Concatenate()([img_stack, state_input])

        output_dense = None
        if hybrid:
            # Add a few dense layers to finish the model
            merged = Dense(self.values.dense_0_units, activation=self.values.activation_function, name='dense_qnn0')(
                merged)
            merged = Dropout(self.values.dense_0_dropout)(merged)
            # different hybrid VQA circuits
            if self.hybrid_type == 0:
                merged = Dense(4, activation=self.values.activation_function, name='dense_pre_out')(merged)
                merged = Dropout(self.values.dense_1_dropout)(merged)
                #VQA layer
                # 4 qubits, 4 input and 4 output
                qlayer = self.vqa_obj.create_two_layer_circuit(trainable=qml_trainable)
                x = qlayer(merged)
                x = Dense(4, activation=self.values.activation_function, name='dense_qnn2')(x)
                output_dense = Dense(self.values.dense_2_units, name='output_qnn')(x)
            elif self.hybrid_type == 1:
                merged = Dense(4, activation=self.values.activation_function, name='dense_pre_out')(merged)
                merged = Dropout(self.values.dense_1_dropout)(merged)
                # VQA layer
                # 6 qubits, 4 input and 2 output
                qlayer = self.vqa_obj.create_custom_v1_circuit(trainable=qml_trainable)
                x = qlayer(merged)
                x = Dense(2, activation=self.values.activation_function, name='dense_qnn2')(x)
                output_dense = Dense(self.values.dense_2_units, name='output_qnn')(x)
            elif self.hybrid_type == 2:
                merged = Dense(4, activation=self.values.activation_function, name='dense_pre_out')(merged)
                merged = Dropout(self.values.dense_1_dropout)(merged)
                # VQA layer
                # 6 qubits, 4 input and 2 output - additional gates compared to hybrid_type 1
                qlayer = self.vqa_obj.create_custom_v2_circuit(trainable=qml_trainable)
                x = qlayer(merged)
                x = Dense(2, activation=self.values.activation_function, name='dense_qnn2')(x)
                output_dense = Dense(self.values.dense_2_units, name='output_qnn')(x)
            elif self.hybrid_type == 3:
                merged = Dense(4, activation=self.values.activation_function, name='dense_pre_out')(merged)
                merged = Dropout(self.values.dense_1_dropout)(merged)
                # VQA layer
                # 4 qubits, 4 input and 2 output
                qlayer = self.vqa_obj.create_custom_v3_circuit(trainable=qml_trainable)
                x = qlayer(merged)
                x = Dense(2, activation=self.values.activation_function, name='dense_qnn2')(x)
                output_dense = Dense(self.values.dense_2_units, name='output_qnn')(x)
            elif self.hybrid_type == 4:
                merged = Dense(4, activation=self.values.activation_function, name='dense_pre_out')(merged)
                merged = Dropout(self.values.dense_1_dropout)(merged)
                # VQA layer
                # 4 qubits, 4 input and 2 output - additional gates compared to hybrid_type 3
                qlayer = self.vqa_obj.create_custom_v4_circuit(trainable=qml_trainable)
                x = qlayer(merged)
                x = Dense(2, activation=self.values.activation_function, name='dense_qnn2')(x)
                output_dense = Dense(self.values.dense_2_units, name='output_qnn')(x)
            elif self.hybrid_type == 5:
                merged = Dense(4, activation=self.values.activation_function, name='dense_pre_out')(merged)
                merged = Dropout(self.values.dense_1_dropout)(merged)
                # VQA layer
                qlayer = self.vqa_obj.create_custom_v5_circuit(trainable=qml_trainable)
                x = qlayer(merged)
                x = Dense(2, activation=self.values.activation_function, name='dense_qnn2')(x)
                output_dense = Dense(self.values.dense_2_units, name='output_qnn')(x)
            elif self.hybrid_type == 6:
                merged = Dense(8, activation=self.values.activation_function, name='dense_pre_out')(merged)
                merged = Dropout(self.values.dense_1_dropout)(merged)
                # VQA layer
                qlayer = self.vqa_obj.create_custom_v6_circuit(trainable=qml_trainable)
                x = qlayer(merged)
                x = Dense(2, activation=self.values.activation_function, name='dense_qnn2')(x)
                output_dense = Dense(self.values.dense_2_units, name='output_qnn')(x)


        else:
            # Add a few dense layers to finish the model
            print("classic model architecture is used")
            merged = Dense(self.values.dense_0_units, activation=self.values.activation_function, name='dense0')(merged)
            merged = Dropout(self.values.dense_0_dropout)(merged)
            merged = Dense(self.values.dense_1_units, activation=self.values.activation_function, name='dense2')(merged)
            merged = Dropout(self.values.dense_1_dropout)(merged)
            output_dense = Dense(self.values.dense_2_units, name='output')(merged)

        self.model = Model(inputs=[pic_input, state_input], outputs=output_dense)
        adam = tf.keras.optimizers.Nadam(lr=self.values.learning_rate, beta_1=self.values.beta_1, beta_2=self.values.beta_2,
                     epsilon=self.values.epsilon)
        self.model.compile(optimizer=adam, loss=self.values.loss_function)

    def create_callback_functions(self):
        plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor=self.values.monitored_quantity, factor=self.values.plateau_factor,
                                             patience=self.values.plateau_patience,
                                             min_lr=self.values.plateau_min_learning_rate,
                                             verbose=self.values.plateau_verbose)
        checkpoint_filepath = os.path.join(self.model_output_folder,
                                           self.MODEL_NAME_FORMAT.format(self.MODEL_NAME_FORMAT_FIRST_PART,
                                                                         self.MODEL_NAME_FORMAT_SECOND_PART))
        check_and_create_dir(checkpoint_filepath)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_best_only=self.values.checkpoint_save_best_only,
                                              verbose=self.values.checkpoint_verbose)
        csv_callback = tf.keras.callbacks.CSVLogger(os.path.join(self.model_output_folder, self.MODEL_LOG_FILE))
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor=self.values.monitored_quantity,
                                                patience=self.values.early_stopping_patience,
                                                min_delta=self.values.early_stopping_min_delta,
                                                verbose=self.values.early_stopping_verbose)

        self.callbacks = [checkpoint_callback]#[plateau_callback, csv_callback, checkpoint_callback, early_stopping_callback]#,
                          #SaveInformationCallback(self.db_connection)]

    def start_with_model_fit(self, epoch=None):
        num_train_examples = self.train_dataset[self.H5_PROPERTY_IMAGE].shape[0]
        num_val_examples = self.val_dataset[self.H5_PROPERTY_IMAGE].shape[0]

        tmp_epoch = epoch
        if not tmp_epoch:
            tmp_epoch = self.values.model_training_epoche
        self.history = self.model.fit_generator(self.train_generator,
                                                steps_per_epoch=num_train_examples // self.batch_size,
                                                epochs=tmp_epoch,
                                                callbacks=self.callbacks, validation_data=self.val_generator,
                                                validation_steps=num_val_examples // self.batch_size,
                                                verbose=self.values.model_training_verbose)

    def predict_result(self, current_image, current_car_state):
        result = self.model.predict([current_image, current_car_state])
        return result