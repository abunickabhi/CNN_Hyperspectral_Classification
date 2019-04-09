"""
DEEP LEARNING Models FOR HSI AVIRIS-NG DATA using Keras Training.
Tensorflow is finally getting as simple as keras
with tf2.0
This script allows the user to run several deep NN models 
using Anand hyperspectral datasets. It is designed to benchmark the classification against
state-of-the-art 1D, 2D and 3D CNNs on various public hyperspectral datasets.

The baseline for this repository is Keras functions and DeepHyperX preprocessing tools that have been implemented
The requirement strictly are:
Python 2.7
Tensorflow 1.1.4
Keras 2.1.2
Sklearn > 0.19
"""
import os
import keras
import argparse
import numpy as np
import datetime
#import custom_datasets
import seaborn as sns
from keras import backend as K
K.set_image_dim_ordering('tf')
from utilsK import *
from custom_datasetsK import dtst
from modelsK import get_model,test, save_model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from JLCallbacks import SaveHistory, TestAndSaveEveryN, AccLossPlotter


dataset_names= 'ip'
#dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]
# Argument parser for CLI interaction

parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default=None, choices=dataset_names,
                    help="Dataset to use.")
parser.add_argument('--model', type=str, default=None,
                    help="Model to train. Available:\n"
                    "baseline (fully connected NN), "
                    "hu (1D CNN), "
                    "hamida (3D CNN + 1D classifier), "
                    "lee (3D FCN), "
                    "chen (3D CNN), "
                    "li (3D CNN), "
                    "he (3D CNN), "
                    "luo (3D CNN), "
                    "sharma (2D CNN), "
                    "boulch (1D semi-supervised CNN), "
                    "liu (3D semi-supervised CNN), "
                    "squeezenet, "
                    "mou (1D RNN)")
parser.add_argument('--folder', type=str, help="Folder where to store the "
                    "datasets (defaults to the current working directory).",
                    default="./Datasets/")
parser.add_argument('--cuda', action='store_true',
                    help="Use CUDA (defaults to false)")
parser.add_argument('--runs', type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument('--restore', type=str, default=None,
                    help="Weights to use for initialization, e.g. a checkpoint")

# Dataset options
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--training_sample', type=int, default=10,
                    help="Percentage of samples to use for training (default: 10%%)")
group_dataset.add_argument('--sampling_mode', type=str, help="Sampling mode"
                    " (random sampling or disjoint, default: random)",
                    default='random')
group_dataset.add_argument('--train_set', type=str, default=None,
                    help="Path to the train ground truth (optional, this "
                    "supersedes the --sampling_mode option)")
group_dataset.add_argument('--test_set', type=str, default=None,
                    help="Path to the test set (optional, by default "
                    "the test_set is the entire ground truth minus the training)")
# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int, help="Training epochs (optional, if"
                    " absent will be set by the model)")
group_train.add_argument('--patch_size', type=int,
                    help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")
group_train.add_argument('--lr', type=float,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--class_balancing', action='store_true',
                    help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--batch_size', type=int,
                    help="Batch size (optional, if absent will be set by the model")
# Test options
group_test = parser.add_argument_group('Test')
group_test.add_argument('--test_stride', type=int, default=1,
                     help="Sliding window step stride during inference (default = 1)")
group_test.add_argument('--inference', type=str, default=None, nargs='?',
                     help="Path to an image on which to run inference.")

# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true',
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true',
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true',
                    help="Random mixes between spectra")

parser.add_argument('--with_exploration', action='store_true',
                    help="See data exploration visualization")
parser.add_argument('--download', type=str, default=None, nargs='+',
                    choices=dataset_names,
                    help="Download the specified datasets and quits.")

args = parser.parse_args()
CUDA = args.cuda
# % of training samples
SAMPLE_PERCENTAGE = args.training_sample / 100
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
INFERENCE = args.inference
TEST_STRIDE = args.test_stride
#DATASET=args.dataset
#mean_spectrums = explore_spectrums(img, gt, LABEL_VALUES,ignored_labels=IGNORED_LABELS)


if CUDA:
    print("Using CUDA")
else:
    print("Not using CUDA, will run on CPU.")


# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS = dtst(FOLDER)
# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]
hyperparams = vars(args)

'''
if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(x):
    return convert_to_color_(x, palette=palette)
def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)
'''
# Instantiate the experiment based on predefined networks
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)


results = []
# run the experiment several times
for run in range(N_RUNS):
    if TRAIN_GT is not None and TEST_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = open_file(TEST_GT)
    elif TRAIN_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = np.copy(gt)
        w, h = test_gt.shape
        test_gt[(train_gt > 0)[:w,:h]] = 0
    elif TEST_GT is not None:
        test_gt = open_file(TEST_GT)
    else:
	# Sample random training spectra
        SAMPLE_PERCENTAGE = 0.1
        train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE)
    print("{} samples selected (over {})".format(np.count_nonzero(train_gt),
                                                 np.count_nonzero(gt)))
    print("Running an experiment with the {} model".format(MODEL),
          "run {}/{}".format(run + 1, N_RUNS))

    if MODEL == 'SGD':
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        class_weight = 'balanced' if CLASS_BALANCING else None
        clf = sklearn.linear_model.SGDClassifier(class_weight=class_weight, learning_rate='optimal', tol=1e-3, average=10)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(scaler.transform(img.reshape(-1, N_BANDS)))
        prediction = prediction.reshape(img.shape[:2])
    else:
        # Neural network
        MODEL = 'nn'
        model,model_name, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
        
        if CLASS_BALANCING:
            weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            hyperparams['weights'] = weights
        # Split train set in train/val
        
        train_gt, val_gt = sample_gt(train_gt, 0.7)
        # Generate the dataset
        train_loader = HyperX(img, train_gt, **hyperparams)
        val_loader = HyperX(img, val_gt, **hyperparams)
        train_steps=train_loader.__len__()
        valid_steps=val_loader.__len__()
        
        if CHECKPOINT is not None:
            model.load_weights(CHECKPOINT)
            
        if model_name=="LiuEtAl" or model_name=="BoulchEtAl":
            if model_name=="LiuEtAl":
                aux_weights=1.0
            else:
                aux_weights=0.1
            model.compile(optimizer=optimizer, loss=loss,loss_weights=[1.0,aux_weights], metrics=['accuracy'])
        else:
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            
        model_dir = './checkpoints/'
        '''
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        '''    
        filename = '/history.txt'
        checkpoint_name = "_epoch{epoch:02d}_{val_acc:.2f}.h5"
        
        try:
            
            checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_acc',save_best_only=True, save_weights_only=True,verbose=0, mode='max')
            cb_hist = SaveHistory(filename)
            plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True)
#            history=model.fit_generator(train_loader, epochs=2,steps_per_epoch = train_loader.__len__(),validation_data=val_loader, validation_steps=val_loader.__len__(),callbacks=[cb_hist,plotter])
            history=model.fit_generator(train_loader, epochs=hyperparams['epoch'], steps_per_epoch = train_steps,validation_data=val_loader, validation_steps=valid_steps,callbacks=[checkpoint,plotter])
        except KeyboardInterrupt:
            
            # Allow the user to stop the training
            pass
#        
#       model.save_weights( model_dir+"/model_squeeze.hdf5")
        probabilities = test(model, img, hyperparams)
        prediction = np.argmax(probabilities, axis=-1)+1



    run_results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=N_CLASSES)
    results.append(run_results)
    show_results(run_results, label_values=LABEL_VALUES)

    #color_prediction = convert_to_color(prediction)
    #display_predictions(color_prediction, gt=convert_to_color(test_gt), caption="Prediction vs. test ground truth")



if N_RUNS > 1:
    "This result shows the accurate way of displaying the model results. The results should be"
    show_results(results,  label_values=LABEL_VALUES, agregated=True)




























































