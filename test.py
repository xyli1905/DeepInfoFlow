import torch 
from model import Model

test = Model(dims = [12, 12, 10, 7, 5, 4, 3, 2, 2], train=True)
ckpt = torch.load('model_epoch_1197.pth')
print(ckpt['model_state_dict'])


# import keras
# import keras.backend as K
# import tensorflow as tf
# import numpy as np
# import pickle

# input_layer = keras.layers.Input((12,))
# clayer = input_layer
# for n in [12, 10, 7, 5, 4, 3, 2]:
#     clayer = keras.layers.Dense(n, 
#                                 activation='tanh',
#                                 kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=1/np.sqrt(float(n)), seed=None),
#                                 bias_initializer='zeros'
#                                )(clayer)
# output_layer = keras.layers.Dense(2, activation='softmax')(clayer)

# model = keras.models.Model(inputs=input_layer, outputs=output_layer)
# sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# model.load_weights("final.h5")

# for layer in model.layers:
#     print(layer)
#     print(layer.get_weights())
