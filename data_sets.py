# <ASSIGNMENT: Generate and load your data sets. Motivate your choices in the docstrings and comments. This file
# contains a suggested structure; you are free to define your own structure, adjust function arguments etc. Don't forget
# to write appropriate tests for your functionality.>

import os
import random
import tensorflow as tf
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Current file marks the root directory
TRAINING_IMAGE_DIR = os.path.join(ROOT_DIR, "data_sets", "training_images")  # Directory for storing training images
TEST_IMAGE_DIR = os.path.join(ROOT_DIR, "data_sets", "test_images")  # Directory for storing test images

LABELS = ['J', 'Q', 'K', 'A']  # Possible card labels 
IMAGE_SIZE = 32 
BATCH_SIZE = 64
ROTATE_MAX_ANGLE = 15

N_TRAIN_IMAGES = 20000
N_TEST_IMAGES = 4000

FONTS = [
    font_manager.findfont(font_manager.FontProperties(family = 'sans-serif', style = 'normal', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'sans-serif', style = 'italic', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'sans-serif', style = 'normal', weight = 'medium')),
    font_manager.findfont(font_manager.FontProperties(family = 'serif', style = 'normal', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'serif', style = 'italic', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'serif', style = 'normal', weight = 'medium')),
]  # True type system fonts


def extract_features(img: Image):
    """
    Convert an image to features that serve as input to the image classifier.

    Arguments
    ---------
    img : Image
        Image to convert to features.

    Returns
    -------
    features : list/matrix/structure of int, int between zero and one
        Extracted features in a format that can be used in the image classifier.
    """

    # <ASSIGNMENT: Implement your feature extraction by converting pixel intensities to features.>
    # features = []
    # for i in range(img.width):
    #     for j in range(img.height):
    #         coordinate = x, y = i, j
    #         features.append(img.getpixel(coordinate)/255)       #append pixel values for each pixel in the image
    features = np.array(img)
    features = features.reshape(1,features.shape[0], features.shape[1], 1).astype('float32') / 255

    return features


def load_data_set(data_dir, n_validation = 0.2):
    """
    Prepare features for the images in data_dir and divide in a training and validation set.

    Parameters
    ----------
    data_dir : str
        Directory of images to load
    n_validation : float between 0 and 1
        Number of images that are assigned for the validation subset 
    """

    features = ImageDataGenerator(rescale=1. / 255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  rotation_range=2*ROTATE_MAX_ANGLE,
                                  horizontal_flip=True,
                                  validation_split=n_validation,
                                  fill_mode='nearest')


    train_gen = features.flow_from_directory(data_dir, color_mode='grayscale',
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                            batch_size=BATCH_SIZE,
                                            class_mode='categorical',
                                            subset='training', shuffle=True)

    val_gen = features.flow_from_directory(data_dir, color_mode='grayscale',
                                          target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                          batch_size=BATCH_SIZE, 
                                          class_mode='categorical',
                                          subset='validation', shuffle=True)
                                          
    # train_ds = tf.data.Dataset.from_generator(lambda: train_gen, output_types=(tf.float32,tf.float32), 
    #                                           output_shapes=([None, IMAGE_SIZE, IMAGE_SIZE, 1],[None, len(LABELS)]))
    # val_ds = tf.data.Dataset.from_generator(lambda: val_gen, output_types=(tf.float32,tf.float32), 
    #                                          output_shapes=([None, IMAGE_SIZE, IMAGE_SIZE, 1],[None, len(LABELS)]))

    return train_gen, val_gen

    # # Extract png files
    # files = os.listdir(data_dir)
    # png_files = []
    # for file in files:
    #     if file.split('.')[-1] == "png":
    #         png_files.append(file)

    # random.shuffle(png_files)  # Shuffled list of the png-file names that are stored in data_dir

    # # <ASSIGNMENT: Load the training and validation set and prepare the features and labels. Use extract_features()
    # # to convert a loaded image (you can load an image with Image.open()) to features that can be processed by your
    # # image classifier. You can extract the original label from the image filename.>
    # training_features = [] # np.empty([32, 32],dtype=np.float32)
    # training_labels = []
    # validation_features = []# np.empty([32, 32],dtype=np.float32)
    # validation_labels = []

    # nr_train = len(png_files)-n_validation
    # png_files_train = png_files[:nr_train]
    # png_files_val = png_files[nr_train:]

    # test_len = len(png_files)   #divide files in train and validation set

    # for file in png_files_train: #append training features and labels to the corresponding lists
    #     # training_features.append(extract_features(Image.open(os.path.join(data_dir, file))))
    #     # np.append(training_features,extract_features(Image.open(os.path.join(data_dir, file))))
    #     training_labels.append(file.split('_')[0])
    
    # for file in png_files_val: #append validation features and labels to the corresponding lists
    #     # validation_features.append(extract_features(Image.open(os.path.join(data_dir, file))))
    #     np.append(validation_features,extract_features(Image.open(os.path.join(data_dir, file))))
    #     validation_labels.append(file.split('_')[0])
    

    # return training_features, training_labels, validation_features, validation_labels


def generate_data_set(n_samples, data_dir):
    """
    Generate n_samples noisy images by using generate_noisy_image(), and store them in data_dir.

    Arguments
    ---------
    n_samples : int
        Number of train/test examples to generate
    data_dir : str in [TRAINING_IMAGE_DIR, TEST_IMAGE_DIR]
        Directory for storing images
    """

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)  # Generate a directory for data set storage, if not already present

    for i in range(n_samples):
        # <ASSIGNMENT: Replace with your implementation. Pick a random rank and convert it to a noisy image through
        # the generate_noisy_image() function below.>
        rank = random.choice(LABELS)
        img = generate_noisy_image(rank, random.uniform(0, 0.3))
        # img.save(f"./{data_dir}/{rank}_{i}.png")   #Original line of code (did not work on my device)
        if not os.path.exists(os.path.join(data_dir,rank)):
            os.makedirs(os.path.join(data_dir,rank))
        img.save(f"{data_dir}/{rank}/{i}.png")     # The filename encodes the original label for training/testing


def generate_noisy_image(rank, noise_level):
    """
    Generate a noisy image with a given noise corruption. This implementation mirrors how the server generates the
    images. However the exact server settings for noise_level and ROTATE_MAX_ANGLE are unknown.
    For the PokerBot assignment you won't need to update this function, but remember to test it.

    Arguments
    ---------
    rank : str in ['J', 'Q', 'K']
        Original card rank.
    noise_level : int between zero and one
        Probability with which a given pixel is randomized.

    Returns
    -------
    noisy_img : Image
        A noisy image representation of the card rank.
    """

    if not 0 <= noise_level <= 1:
        raise ValueError(f"Invalid noise level: {noise_level}, value must be between zero and one")
    if rank not in LABELS:
        raise ValueError(f"Invalid card rank: {rank}")

    # Create rank image from text
    font = ImageFont.truetype(random.choice(FONTS), size = IMAGE_SIZE - 6)  # Pick a random font
    img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), color = 255)
    draw = ImageDraw.Draw(img)
    (text_width, text_height) = draw.textsize(rank, font = font)  # Extract text size
    draw.text(((IMAGE_SIZE - text_width) / 2, (IMAGE_SIZE - text_height) / 2 - 4), rank, fill = 0, font = font)

    # Random rotate transformation
    img = img.rotate(random.uniform(-ROTATE_MAX_ANGLE, ROTATE_MAX_ANGLE), expand = False, fillcolor = '#FFFFFF')
    pixels = list(img.getdata())  # Extract image pixels

    # Introduce random noise
    for (i, _) in enumerate(pixels):
        if random.random() <= noise_level:
            pixels[i] = random.randint(0, 255)  # Replace a chosen pixel with a random intensity

    # Save noisy image
    noisy_img = Image.new('L', img.size)
    noisy_img.putdata(pixels)

    return noisy_img

if __name__=='__main__':
    generate_data_set(N_TRAIN_IMAGES,TRAINING_IMAGE_DIR)
    generate_data_set(N_TEST_IMAGES, TEST_IMAGE_DIR)