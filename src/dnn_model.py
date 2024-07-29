import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.layers import Dense
from scikeras.wrappers import KerasClassifier

tf.get_logger().setLevel('ERROR')

# def limit_tf_gpu_emory(memory_limit=2048):
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#
#     if gpus:
#         try:
#             # Set memory limit (e.g., 2GB)
#             memory_limit = memory_limit  # in MB
#             for gpu in gpus:
#                 tf.config.experimental.set_virtual_device_configuration(
#                     gpu,
#                     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
#                 )
#         except RuntimeError as e:
#             print(e)


# disable gpu
tf.config.set_visible_devices([], 'GPU')

# limit gpu memory
# limit_tf_gpu_emory(round(4*1024))


# Base Classifier
class DNNClassifier(KerasClassifier):
    def __init__(self, num_classes, num_features, random_state=42, hidden_layers_size="16, 8",
                 activation_function='relu',
                 loss_function='categorical_crossentropy', nn_optimizer='adam', epochs=150, batch_size=10, **kwargs):

        super().__init__(num_classes=num_classes, num_features=num_features,
                         hidden_layers_size=hidden_layers_size, activation_function=activation_function,
                         loss_function=loss_function, nn_optimizer=nn_optimizer, epochs=epochs, batch_size=batch_size)

        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.epochs = epochs
        self.batch_size = batch_size

        self.__enabled_codified_predict = False

    def _keras_build_fn(self, num_classes, num_features, hidden_layers_size, activation_function, loss_function,
                        nn_optimizer):

        # Creating a Keras Model
        model = Sequential()
        hidden_layers_size = [int(layer_size) for layer_size in hidden_layers_size.split(', ')]

        model.add(Dense(hidden_layers_size[0], input_shape=(num_features,), activation=activation_function))
        for layer_size in hidden_layers_size[1:]:
            if layer_size is not None:
                model.add(Dense(layer_size, activation=activation_function))
        model.add(Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid'))
        # Compile the Keras model
        model.compile(loss=loss_function, optimizer=nn_optimizer,
                      metrics=['accuracy'])  # the model itself does not depend on the metric

        return model

    def set_enabled_codified_predict(self, enabled):
        self.__enabled_codified_predict = enabled

    def predict(self, X, **kwargs):
        # print("Predicting...")
        kwargs['verbose'] = 0

        y_pred = super().predict(X, **kwargs)
        if self.__enabled_codified_predict:
            y_pred = tf.argmax(y_pred, axis=1)

        return y_pred

    def predict_proba(self, X, **kwargs):
        kwargs['verbose'] = 0
        y_proba = super().predict(X, **kwargs)
        return y_proba

    def fit(self, X, y, **kwargs):

        # print("Fitting the model...")
        kwargs['epochs'] = self.epochs
        kwargs['batch_size'] = self.batch_size
        kwargs['verbose'] = 0

        return super().fit(X, y, **kwargs)


# # DNN using PyTorch
#
# import torch
# from skorch import NeuralNetClassifier
# from torch import nn, optim, Tensor
# from torch.utils.data import Subset
# import numpy as np
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# class MyModule(nn.Module):
#     def __get_activation_function(self, name):
#         if name == 'sigmoid':
#             return nn.Sigmoid()
#         elif name == 'tanh':
#             return nn.Tanh()
#         elif name == 'relu':
#             return nn.ReLU()
#         elif name == 'silu':
#             return nn.SiLU()
#         else:
#             raise ValueError(f'Unknown activation function: {name}')
#
#     def __init__(self, num_features, hidden_layers_size, num_classes, activation_function):
#         super(MyModule, self).__init__()
#
#         # Parse the hidden_layers_size string into a list of integers
#         hidden_layers_size = [int(layer_size) for layer_size in hidden_layers_size.split(', ')]
#
#         # Create a list to hold the layers
#         self.layers = nn.ModuleList()
#
#         # Add the first layer
#         self.layers.append(nn.Linear(num_features, hidden_layers_size[0]))
#         self.layers.append(self.__get_activation_function(activation_function))
#
#         last_layer_size = hidden_layers_size[0]
#
#         # Add more hidden layers if they exist
#         for layer_size in hidden_layers_size[1:]:
#             self.layers.append(nn.Linear(last_layer_size, layer_size))
#             self.layers.append(self.__get_activation_function(activation_function))
#             last_layer_size = layer_size
#
#         # Add the output layer
#         self.layers.append(nn.Linear(last_layer_size, num_classes))
#         if num_classes > 2:
#             self.layers.append(nn.Softmax(dim=1))
#         else:
#             self.layers.append(nn.Sigmoid())
#
#     def forward(self, X):
#         for layer in self.layers:
#             X = layer(X)
#         return X
#
#
# class DNNClassifierV2(NeuralNetClassifier):
#     def __get_optimizer(self, name):
#         if not isinstance(name, str):
#             return name
#
#         mapping = {
#             'adam': optim.Adam,
#             'rmsprop': optim.RMSprop,
#             'nadam': optim.NAdam,
#         }
#
#         optimizer_class = mapping.get(name.lower())
#         if optimizer_class is None:
#             raise ValueError(f'Unknown optimizer: {name}')
#
#         return optimizer_class
#
#     def __get_loss_function(self, name):
#         if not isinstance(name, str):
#             return name
#
#         if name == 'categorical_crossentropy':
#             return nn.CrossEntropyLoss()
#         else:
#             raise ValueError(f'Unknown loss function: {name}')
#
#     def __init__(self, module__num_classes=0, module__num_features=0, random_state=42,
#                  module__hidden_layers_size="16, 8",
#                  module__activation_function='relu',
#                  criterion='categorical_crossentropy', optimizer='adam', max_epochs=150, batch_size=10, **kwargs):
#
#         is_not_a_clone = kwargs == {}
#
#         if is_not_a_clone:
#             self.module__num_features = module__num_features
#             self.module__num_classes = module__num_classes
#             self.module__hidden_layers_size = module__hidden_layers_size
#             self.module__activation_function = module__activation_function
#
#             self.max_epochs = max_epochs
#             self.module = MyModule
#             self.batch_size = batch_size
#             self.criterion = criterion
#
#             super(DNNClassifierV2, self).__init__(
#                 module=MyModule,
#                 batch_size=self.batch_size,
#                 criterion=self.__get_loss_function(self.criterion),
#                 optimizer=self.__get_optimizer(optimizer),
#                 max_epochs=self.max_epochs,
#                 module__num_features=self.module__num_features,
#                 module__num_classes=self.module__num_classes,
#                 module__hidden_layers_size=self.module__hidden_layers_size,
#                 module__activation_function=self.module__activation_function,
#                 verbose=False,  # To hide the training process logs
#                 device=device  # Set the device to GPU if available
#             )
#
#         else:
#             self.module__num_features = kwargs.get('module__num_features', module__num_features)
#             self.module__num_classes = kwargs.get('module__num_classes', module__num_classes)
#             self.module__hidden_layers_size = kwargs.get('module__hidden_layers_size', module__hidden_layers_size)
#             self.module__activation_function = kwargs.get('module__activation_function', module__activation_function)
#
#             self.batch_size = kwargs.get('batch_size', batch_size)
#             self.criterion = kwargs.get('criterion', criterion)
#             self.optimizer = kwargs.get('optimizer', optimizer)
#             self.max_epochs = kwargs.get('max_epochs', max_epochs)
#
#             super(DNNClassifierV2, self).__init__(
#                 batch_size=self.batch_size,
#                 criterion=self.__get_loss_function(self.criterion),
#                 optimizer=self.__get_optimizer(self.optimizer),
#                 max_epochs=self.max_epochs,
#                 module__hidden_layers_size=self.module__hidden_layers_size,
#                 module__activation_function=self.module__activation_function,
#                 device=device,  # Set the device to GPU if available
#                 **kwargs
#             )
#
#         np.random.seed(random_state)
#         torch.manual_seed(random_state)
#
#     def fit(self, X, y, **fit_params):
#         X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32).to(device)
#         y_tensor = torch.tensor(y.to_numpy(), dtype=torch.long).to(device)
#         super().fit(X_tensor, y_tensor, **fit_params)
#
#     def predict(self, X):
#         if isinstance(X, Subset):
#             return super().predict(X)
#         X_tensor = torch.tensor(X if isinstance(X, np.ndarray) else X.to_numpy(), dtype=torch.float32).to(device)
#         return super().predict(X_tensor)
#
#     def predict_proba(self, X):
#         if isinstance(X, Tensor):
#             return super().predict_proba(X)
#         if isinstance(X, Subset):
#             return super().predict_proba(X)
#         X_tensor = torch.tensor(X if isinstance(X, np.ndarray) else X.to_numpy(), dtype=torch.float32).to(device)
#         return super().predict_proba(X_tensor)
#
#     def set_params(self, **parameters):
#         self.module__hidden_layers_size = parameters.get('module__hidden_layers_size', self.module__hidden_layers_size)
#         self.module__activation_function = parameters.get('module__activation_function', self.module__activation_function)
#         self.optimizer = self.__get_optimizer(parameters.get('optimizer', self.optimizer))
#         self.max_epochs = parameters.get('max_epochs', self.max_epochs)
#         self.batch_size = int(parameters.get('batch_size', self.batch_size))
#         return {}

