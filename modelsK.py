""" 
LIST OF MODELS FILE

This file contains all the models that need to be implemented for classification purpose.
For any Pytorch model that has benchmarked parameters, I will converting it to Keras
sequential or non-sequential model by available model converters.

"""

import os
import sys
import keras
import datetime
import numpy as np
from keras import layers
from keras.layers import *
from sklearn.externals import joblib
from keras.models import Sequential
from keras.models import Model
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras import optimizers, regularizers
from utilsK import grouper, sliding_window, count_sliding_window,camel_to_snake

name = 'hu'
def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with thr hyperparameters
    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: Keras network
        optimizer: Keras optimizer
        criterion: Keras loss Function
        kwargs: hyperparameters with sane defaults
    """
#    cuda = kwargs.setdefault('cuda', False)
    n_classes = kwargs['n_classes']
    #name = kwargs['dataset']
    n_bands = kwargs['n_bands']
    weights = np.ones(n_classes)
#    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = kwargs.setdefault('weights', weights)
    model_name=None
    if name == 'nn':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True

        #model,model_name = Baseline(n_bands, n_classes, kwargs.setdefault('dropout', True))
        model,model_name = _1April(n_bands, n_classes, kwargs.setdefault('dropout', True))
        lr = kwargs.setdefault('learning_rate', 0.0001)
        optimizer = optimizers.Adam(lr=lr)
#       optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = 'categorical_crossentropy'
        kwargs.setdefault('epoch', 100)
        kwargs.setdefault('batch_size', 100)
    elif name == 'hu':
        kwargs.setdefault('patch_size', 1)
        kwargs.setdefault('epoch', 100)
        kwargs.setdefault('batch_size', 100)
        center_pixel = True
#        input_channels=((kwargs['batch_size'],n_bands))
        model, model_name = HuEtAl.build(n_bands, n_classes)
        lr = kwargs.setdefault('learning_rate', 0.01)
        optimizer = optimizers.Adam(lr=lr)
        criterion = 'categorical_crossentropy'
    elif name == 'hamida':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model,model_name = HamidaEtAl.build(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.01)
        optimizer = optimizers.SGD(lr=lr, decay=0.0005)
        kwargs.setdefault('batch_size', 100)
        criterion = 'categorical_crossentropy'
    elif name == 'lee':
        kwargs.setdefault('epoch', 200)
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model, model_name = LeeEtAl.build(n_bands, n_classes)
        lr = kwargs.setdefault('learning_rate', 0.001)
        optimizer = optimizers.Adam(lr=lr)
        criterion = 'categorical_crossentropy'

    elif name == 'chen':
        patch_size = kwargs.setdefault('patch_size', 27)
        center_pixel = True
        model,model_name = ChenEtAl.build(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.003)
        optimizer = optimizers.SGD(lr=lr)
        criterion = 'categorical_crossentropy'
        kwargs.setdefault('epoch', 400)
        kwargs.setdefault('batch_size', 100)
    elif name == 'li':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model,model_name = LiEtAl.build(n_bands, n_classes, n_planes=16, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.01)
        optimizer = optimizers.SGD(lr=lr, momentum=0.9, decay=0.0005)
        epoch = kwargs.setdefault('epoch', 200)
        criterion = 'categorical_crossentropy'
    elif name == 'he':
        kwargs.setdefault('patch_size', 7)
        kwargs.setdefault('batch_size', 40)
        lr = kwargs.setdefault('learning_rate', 0.01)
        center_pixel = True
        model,model_name = HeEtAl.build(n_bands, n_classes, patch_size=kwargs['patch_size'])
        # For Adagrad, we need to load the model on GPU before creating the optimizer
      
        optimizer = optimizers.Adagrad(lr=lr, decay=0.01)
        criterion = 'categorical_crossentropy'
    elif name == 'luo':
        # All  the  experiments  are  settled  by  the  learning  rate  of  0.1,
        # the  decay  term  of  0.09  and  batch  size  of  100.
        kwargs.setdefault('patch_size', 3)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('learning_rate', 0.1)
        center_pixel = True
        model,model_name = LuoEtAl.build(n_bands, n_classes, patch_size=kwargs['patch_size'])
        optimizer = optimizers.SGD(lr=lr, decay=0.09)
        criterion = 'categorical_crossentropy'
    elif name == 'sharma':
        # We train our S-CNN from scratch using stochastic gradient descent with
        # momentum set to 0.9, weight decay of 0.0005, and with a batch size
        # of 60.  We initialize an equal learning rate for all trainable layers
        # to 0.05, which is manually decreased by a factor of 10 when the validation
        # error stopped decreasing. Prior to the termination the learning rate was
        # reduced two times at 15th and 25th epoch. [...]
        # We trained the network for 30 epochs
        kwargs.setdefault('batch_size', 60)
        epoch = kwargs.setdefault('epoch', 30)
        lr = kwargs.setdefault('lr', 0.05)
        center_pixel = True
        # We assume patch_size = 64
        kwargs.setdefault('patch_size', 64)
        model,model_name = SharmaEtAl.build(n_bands, n_classes, patch_size=kwargs['patch_size'])
        optimizer = optimizers.SGD(lr=lr,decay=0.0005)
        criterion = 'categorical_crossentropy'
    elif name == 'liu':
        kwargs['supervision'] = 'semi'
        # "The learning rate is set to 0.001 empirically. The number of epochs is set to be 40."
        kwargs.setdefault('epoch', 40)
        lr = kwargs.setdefault('lr', 0.001)
        center_pixel = True 
        patch_size = kwargs.setdefault('patch_size', 9)
        model,model_name = LiuEtAl.build(n_bands, n_classes, patch_size)
        optimizer = optimizers.SGD(lr=lr)
        criterion = ['categorical_crossentropy','mean_squared_error']#weighted_loss(1.0)
#        K.mean(K.square(rec, squeeze_all(data[:,:,:,patch_size//2,patch_size//2])))
#        kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1))
    elif name == 'boulch':
        kwargs['supervision'] = 'semi'
        kwargs.setdefault('patch_size', 1)
        kwargs.setdefault('epoch', 100)
        lr = kwargs.setdefault('lr', 0.001)
        center_pixel = True
        model,model_name = BoulchEtAl.build(n_bands, n_classes)
        optimizer = optimizers.SGD( lr=lr)
        criterion =['categorical_crossentropy','mean_squared_error'] #weighted_loss(0.1)
    elif name == 'mou':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        kwargs.setdefault('epoch', 100)
        # "The RNN was trained with the Adadelta algorithm [...] We made use of a
        # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
        # default of  0.002 to  train the  network"
        lr = kwargs.setdefault('lr', 1.0)
        model,model_name = MouEtAl.build(n_bands, n_classes)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
#        model = model.to(device)
        optimizer = optimizers.Adadelta(lr=lr)
        criterion = 'categorical_crossentropy'
    elif name=='squeezenet':
        kwargs.setdefault('patch_size', 3)
        kwargs.setdefault('batch_size', 40)
        kwargs.setdefault('epoch', 100)
        lr = kwargs.setdefault('learning_rate',0.5)
        center_pixel = True
        model,model_name = Squeezenet().build(n_bands,n_classes, patch_size=kwargs['patch_size'])
        optimizer = optimizers.Adadelta(lr=lr)
        criterion = 'categorical_crossentropy'
    else:
        raise KeyError("{} model is unknown.".format(name))

    epoch = kwargs.setdefault('epoch', 100)
    #kwargs.setdefault('scheduler', None)
    kwargs.setdefault('batch_size', 100)
    kwargs.setdefault('dataset', None)
    kwargs.setdefault('supervision', 'full')
    kwargs.setdefault('flip_augmentation', False)
    kwargs.setdefault('radiation_augmentation', False)
    kwargs.setdefault('mixture_augmentation', False)
    kwargs['center_pixel'] = center_pixel
    return model,model_name, optimizer, criterion, kwargs


def Baseline(input_shape, num_classes, dropout=True):
    model_name='Baseline'
    model = Sequential()
    model.add(Dense(2048, input_dim=input_shape, activation='relu'))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Dense(4096, activation='relu'))
    if dropout:
        model.add(Dropout(0.2))    
    model.add(Dense(2048, activation='relu'))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    #model.add('softmax')
    model.add(Dense(num_classes, activation='softmax'))
    return model,model_name



def _27March(input_shape, num_classes, dropout=True):
    """This model is written on keras and the whole repository contains the files of various 
    models that have been implemented on open HSI datasets.
    This function will operate on a 19 class dataset which is very sparse.
    Since there is sparsity the network has to be flexible hence the dropout.
    """
    model_name='_27March'
    model = Sequential()
    model.add(Dense(204, input_dim=input_shape, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(409, activation='relu'))
    model.add(Dropout(0.5))    
    model.add(Dense(204, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    return model,model_name


from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

def _1April(input_shape, num_classes, dropout=True):
    """
    This function will operate on a 19 class dataset which is very sparse so the basic idea is to divide it into training 
    and testing mask.
    The model tensor has to be flattened from the training starts.
    
    """
    model = Sequential()
    model.add(Conv1D(100, kernel_size=(10),activation='linear',input_dim=input_shape,padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(64, (5, 5), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(64, (5, 5), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model,model_name

def save_model(model, model_name, dataset_name, **kwargs):
     model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
     if not os.path.isdir(model_dir):
         os.makedirs(model_dir)
         filename = str(datetime.datetime.now())
         #tqdm.write("Saving model params in {}".format(filename))
         joblib.dump(model, model_dir + filename + '.pkl')


def test(model, img, hyperparams):
    """
    Test a model on a specific image
    """
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size = hyperparams['batch_size']
    n_classes = hyperparams['n_classes']

    kwargs = {'step': hyperparams['test_stride'], 'window_size': (patch_size, patch_size)}
#    probs = np.zeros(img.shape[:2])
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in grouper(batch_size, sliding_window(img, **kwargs)):
        if patch_size == 1:
            data = [b[0][0, 0] for b in batch]
            data = np.copy(data)
        else:
            data = [b[0] for b in batch]
            data = np.copy(data)
#            print data.shape
            data = data.transpose(0, 3, 1, 2)
            data=np.expand_dims(data,axis=1)
#            data = data.unsqueeze(1)

        indices = [b[1:] for b in batch]
        output=model.predict(data)
        if isinstance(output, list):
            output = output[0]

        for (x, y, w, h), out in zip(indices, output):
            if center_pixel:
                probs[x + w // 2, y + h // 2] += out
            else:
                probs[x:x + w, y:y + h] += out
    return probs

