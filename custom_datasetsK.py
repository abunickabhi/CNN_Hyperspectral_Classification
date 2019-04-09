"""This file contains the loader function for the open datasets and the analysis that can be
performed on them after they are called into the model file."""


import os
import keras
import spectral
import numpy as np
from scipy.io import loadmat 
from utilsK import open_file
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.utils import np_utils

folder='/home/abhi_daiict/Desktop/Hyperspectral_dataset'
CUSTOM_DATASETS_CONFIG = {
         'anand': {
            'img': 'Aviris_Anand.tif',
            'gt': 'Aviris_Anand_gt.tif',
            'download': False,
            'loader': lambda folder: dtst(folder)
            }
    }


def dtst(folder):
    '''
    img = open_file('/home/abhi/Desktop/Hyperspectral Datasets/mat_files/Indian_pines_corrected.mat)[:,:,:-2]
    gt = open_file('/home/abhi_daiict/Desktop/Hyperspectral_dataset/Aviris_Anand_gt.tif')
     
    img=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/mat_files/Indian_pines_corrected.mat')
    img=img['indian_pines_corrected']
    imgt=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/mat_files/Indian_pines_gt.mat')
    gt=imgt['indian_pines_gt']
        
    img=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/mat_files/Salinas.mat')
    img=img['salinas']
    imgt=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/mat_files/Salinas_gt.mat')
    gt=imgt['salinas_gt']
    '''
    img=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/mat_files/Pavia.mat')
    img=img['pavia']
    imgt=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/mat_files/Pavia_gt.mat')
    gt=imgt['pavia_gt']
        
    label_values = ["Unclassified",
                        "Grass",
                        "Baresoil",
                        "Plough Field",
                        "Rail",
                        "Maize",
                        "Building",
                        "Trees",
                        "Bricks",
                        "Water",
                        "Pigeon Pea",
                        "Building Conc",
                        "Wheat",
                        "Mustard",
                        "Road",
                        "Lin Seeds",
                        "Brinjal",
                        "Tobacco",
                        "Amla",
                        "Caster",
                        ]
    ignored_labels = [0]
    
    '''
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]
    '''
    # For Pavia dataset
    label_values = ['Undefined','Water',	
                       'Trees',
	                    'Asphalt',	
	                    'Self-Blocking Bricks',	
	                    'Bitumen',	
	                    'Tiles',	
	                    'Shadows',	
	                    'Meadows',	
	                    'Bare Soil']
    '''
        #for salinas dataset
        label_values = ["Undefined",'Brocoli_green_weeds_1',	
	                      'Brocoli_green_weeds_2'	,
	                       'Fallow',
	                       'Fallow_rough_plow',
                         'Fallow_smooth',
	                      'Stubble',	
                         'Celery',	
	                     'Grapes_untrained',	
	                     'Soil_vinyard_develop',	
	                     'Corn_senesced_green_weeds',	
	                      'Lettuce_romaine_4wk',	
	                     'Lettuce_romaine_5wk',	
	                     'Lettuce_romaine_6wk',	
	                     'Lettuce_romaine_7wk',
	                     'Vinyard_untrained'	,
	                     'Vinyard_vertical_trellis']        
    '''
    
    '''
    palette = dict((
    (0, (0,   0,   0)),  # Unclassified
    (1, (107,   0,   0)),  # Grass
    (2, (255, 23,   38)),  # Bare Soil
    (3, (154,   119, 96)),  # plough Field
    (4, (17, 128, 0)),  # rail
    (5, (42,35,255)),  # Maize
    (6, (146,   239, 48)),  # Building
    (7, (176, 48, 38))))
    '''
    return img,gt,label_values,ignored_labels

def load_data(data, gt,**hyperparams):
    patch=[]
    patch_label=[]
    n_classes=hyperparams['n_classes']
    patch_size = hyperparams['patch_size']
    ignored_labels = set(hyperparams['ignored_labels'])
    center_pixel = hyperparams['center_pixel']
    supervision = hyperparams['supervision']
    # Fully supervised : use all pixels with label not ignored
    if supervision == 'full':
        mask = np.ones_like(gt)
        for l in ignored_labels:
            mask[gt == l] = 0
    # Semi-supervised : use all pixels, except padding
    elif supervision == 'semi':
        mask = np.ones_like(gt)
    x_pos, y_pos = np.nonzero(mask)
    p = patch_size // 2
    indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
     
    for i in range(len(indices)):
        x, y = indices[i]
        x1, y1 = x - patch_size // 2, y - patch_size // 2
        x2, y2 = x1 + patch_size, y1 + patch_size

        xx = data[x1:x2, y1:y2]
        yy = gt[x1:x2, y1:y2]
        print(xx.shape)

#                    if self.flip_augmentation and self.patch_size > 1:
#                        # Perform data augmentation (only on 2D patches)
#                        data, label = self.flip(data, label)
#                    if self.radiation_augmentation and np.random.random() < 0.1:
#                            data = self.radiation_noise(data)
#                    if self.mixture_augmentation and np.random.random() < 0.2:
#                            data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        xx = np.asarray(np.copy(xx).transpose((2, 0, 1)), dtype='float32')
        yy = np.asarray(np.copy(yy), dtype='int64')

        # Extract the center label if needed
        if center_pixel and patch_size > 1:
            yy = yy[patch_size // 2, patch_size // 2]
        
        # Remove unused dimensions when we work with invidual spectrums
        elif patch_size == 1:
            xx = xx[:, 0, 0]
            yy = yy[0, 0]

        # Add a fourth dimension for 3D CNN
        if patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            xx = xx.reshape(1,xx.shape[0], xx.shape[1], xx.shape[2])
#            yy = yy.transpose((2, 0, 1))
            
        patch.append(xx)
        
        patch_label.append(yy)
        
    X_test=np.asarray(patch) 
    y_test=np.asarray(patch_label)
    if supervision == 'semi':
        data_loss=np.asarray(patch)[:,:,:,patch_size//2,patch_size//2].squeeze() 
        y_true=keras.utils.to_categorical(y_test-1, n_classes)
        
        target=[y_true,data_loss]
    else:
        if not center_pixel and patch_size > 1:
            target=keras.utils.to_categorical(y_test-1, n_classes).transpose(0,3,1,2)
            
        else:
            target=keras.utils.to_categorical(y_test-1, n_classes)
    
    
    
    return X_test, target
    

def flip(*arrays):
    horizontal = np.random.random() > 0.5
    vertical = np.random.random() > 0.5
    if horizontal:
        arrays = [np.fliplr(arr) for arr in arrays]
    if vertical:
        arrays = [np.flipud(arr) for arr in arrays]
    return arrays

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

def mixture_noise(indices, labels,ignored_labels, data, label, beta=1/25):
    alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    data2 = np.zeros_like(data)
    for  idx, value in np.ndenumerate(label):
        if value not in ignored_labels:
            l_indices = np.nonzero(labels == value)[0]
            l_indice = np.random.choice(l_indices)
            assert(labels[l_indice] == value)
            x, y = indices[l_indice]
            data2[idx] = data[x,y]
    return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise



def batch_iter(data, gt, shuffle, **hyperparams):
    name = hyperparams['dataset']
    n_classes=hyperparams['n_classes']
    patch_size = hyperparams['patch_size']
    batch_size=hyperparams['batch_size']
    ignored_labels = set(hyperparams['ignored_labels'])
#    flip_augmentation = hyperparams['flip_augmentation']
#    radiation_augmentation = hyperparams['radiation_augmentation'] 
#    mixture_augmentation = hyperparams['mixture_augmentation'] 
    center_pixel = hyperparams['center_pixel']
    supervision = hyperparams['supervision']
    # Fully supervised : use all pixels with label not ignored
    if supervision == 'full':
        mask = np.ones_like(gt)
        for l in ignored_labels:
            mask[gt == l] = 0
    # Semi-supervised : use all pixels, except padding
    elif supervision == 'semi':
        mask = np.ones_like(gt)
    x_pos, y_pos = np.nonzero(mask)
    p = patch_size // 2
    indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
    labels = [gt[x,y] for x,y in indices]
#    np.random.shuffle(indices)
    num_batches_per_epoch = int((len(indices) - 1) / batch_size) + 1

    def data_generator():
        data_size = len(indices)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                np.random.shuffle(indices)
            
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                patch=[]
                patch_label=[]
                indexes=indices[start_index:end_index] 
                for i in range(len(indexes)):
                    x, y = indexes[i]
                    x1, y1 = x - patch_size // 2, y - patch_size // 2
                    x2, y2 = x1 + patch_size, y1 + patch_size
            
                    xx = data[x1:x2, y1:y2]
                    yy = gt[x1:x2, y1:y2]
            
#                    if self.flip_augmentation and self.patch_size > 1:
#                        # Perform data augmentation (only on 2D patches)
#                        data, label = self.flip(data, label)
#                    if self.radiation_augmentation and np.random.random() < 0.1:
#                            data = self.radiation_noise(data)
#                    if self.mixture_augmentation and np.random.random() < 0.2:
#                            data = self.mixture_noise(data, label)
            
                    # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
                    xx = np.asarray(np.copy(xx).transpose((2, 0, 1)), dtype='float32')
                    yy = np.asarray(np.copy(yy), dtype='int64')
            
                    # Extract the center label if needed
                    if center_pixel and patch_size > 1:
                        yy = yy[patch_size // 2, patch_size // 2]
                    # Remove unused dimensions when we work with invidual spectrums
                    elif patch_size == 1:
                        xx = xx[:, 0, 0]
                        yy = yy[0, 0]
            
                    # Add a fourth dimension for 3D CNN
                    if patch_size > 1:
                        # Make 4D data ((Batch x) Planes x Channels x Width x Height)
                        xx = xx.reshape(1,xx.shape[0], xx.shape[1], xx.shape[2])
                    patch.append(xx)
                    patch_label.append(yy)
                    
                X=np.asarray(patch) 
                y=np.asarray(patch_label)
                
                if supervision == 'semi':
                    if patch_size>1 :
                        data_loss=np.asarray(patch)[:,:,:,patch_size//2,patch_size//2].squeeze() 
                    else:
                        data_loss=np.asarray(patch).squeeze() 
                    y_true=keras.utils.to_categorical(y-1, n_classes)
                    target=[y_true,data_loss]
                else:
                    if not center_pixel and patch_size > 1:
                        target=keras.utils.to_categorical(y-1, n_classes).transpose(0,3,1,2)
                    else:
                        target=keras.utils.to_categorical(y-1, n_classes)
    

                yield X, target

    return num_batches_per_epoch, data_generator()



class HyperX(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, gt, **hyperparams):
        'Initialization'
        self.data = data
        self.label = gt
        self.n_classes=hyperparams['n_classes']
        self.batch_size=hyperparams['batch_size']
        self.name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation'] 
        self.mixture_augmentation = hyperparams['mixture_augmentation'] 
        self.center_pixel = hyperparams['center_pixel']
        self.supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if self.supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif self.supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]
         
        self.on_epoch_end()

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for  idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int((len(self.indices) - 1) / self.batch_size) + 1

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X,y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.indices)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        patch=[]
        patch_label=[]
        for i in range(len(indexes)):
            x, y = indexes[i]
            x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
            x2, y2 = x1 + self.patch_size, y1 + self.patch_size
    
            data = self.data[x1:x2, y1:y2]
            label = self.label[x1:x2, y1:y2]
    
            if self.flip_augmentation and self.patch_size > 1:
                # Perform data augmentation (only on 2D patches)
                data, label = self.flip(data, label)
            if self.radiation_augmentation and np.random.random() < 0.1:
                    data = self.radiation_noise(data)
            if self.mixture_augmentation and np.random.random() < 0.2:
                    data = self.mixture_noise(data, label)
    
            # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
            data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
            label = np.asarray(np.copy(label), dtype='int64')
    
            # Extract the center label if needed
            if self.center_pixel and self.patch_size > 1:
                label = label[self.patch_size // 2, self.patch_size // 2]
            # Remove unused dimensions when we work with invidual spectrums
            elif self.patch_size == 1:
                data = data[:, 0, 0]
                label = label[0, 0]
    
            # Add a fourth dimension for 3D CNN
            if self.patch_size > 1:
                # Make 4D data ((Batch x) Planes x Channels x Width x Height)
                data = data.reshape(1,data.shape[0], data.shape[1], data.shape[2])
#                label = label.transpose((2, 0, 1))

            patch.append(data)
            patch_label.append(label)
        
        X=np.asarray(patch) 
        y=np.asarray(patch_label)
#        print X.shape,y.shape
        
        if self.supervision == 'semi':
            if self.patch_size>1 :
                data_loss=np.asarray(patch)[:,:,:,self.patch_size//2,self.patch_size//2].squeeze() 
            else:
                data_loss=np.asarray(patch).squeeze() 
            
            y_true=keras.utils.to_categorical(y-1, self.n_classes)
            target=[y_true,data_loss]
        else:
#            if not self.center_pixel and self.patch_size > 1:
#                target=keras.utils.to_categorical(y-1, self.n_classes).transpose(0,3,1,2)
#            else:
            target=keras.utils.to_categorical(y-1, self.n_classes)
            
        return X, target




