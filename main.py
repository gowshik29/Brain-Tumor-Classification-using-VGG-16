import numpy as np 
from tqdm import tqdm
import cv2
import os
import shutil
import itertools
import imutils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
import plotly.graph_objects as go
import splitfolders

# Input path to dataset and output directory
IMG_PATH = 'C:/Users/AIGow/OneDrive/Desktop/Python/Brain/brain_tumor_dataset'
OUTPUT_PATH = 'output/'

# Split dataset into train, val, test (80%, 10%, 10%)
splitfolders.ratio(IMG_PATH, output=OUTPUT_PATH, seed=42, ratio=(0.8, 0.1, 0.1))

def load_data(dir_path, img_size=(100,100)):
    X = []
    y = []
    labels = {}
    label_index = 0

    for class_name in sorted(os.listdir(dir_path)):
        class_path = os.path.join(dir_path, class_name)

        if not os.path.isdir(class_path):
            continue
        labels[label_index] = class_name

        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)

            if img_file.startswith('.'):
                continue

            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(label_index)
        label_index += 1
    X = np.array(X, dtype="float32")
    y = np.array(y)

    return X, y, labels

TRAIN_DIR = 'C:/Users/AIGow/OneDrive/Desktop/Python/Brain/output/train'
TEST_DIR = 'C:/Users/AIGow/OneDrive/Desktop/Python/Brain/output/test'
VAL_DIR = 'C:/Users/AIGow/OneDrive/Desktop/Python/Brain/output/val'
IMG_SIZE = (224, 224)

# Use predefined function to load the image data into workspace
X_train, y_train, labels = load_data(TRAIN_DIR, IMG_SIZE)
X_test, y_test, _ = load_data(TEST_DIR, IMG_SIZE)
X_val, y_val, _ = load_data(VAL_DIR, IMG_SIZE)

y = dict()
y[0] = []
y[1] = []

for set_name in (y_train, y_val, y_test):
    y[0].append(np.sum(set_name == 0))
    y[1].append(np.sum(set_name == 1))

trace0 = go.Bar(
    x=['Train Set', 'Validation Set', 'Test Set'],
    y=y[0],
    name='No',
    marker=dict(color='#33cc33'),
    opacity=0.7
)
trace1 = go.Bar(
    x=['Train Set', 'Validation Set', 'Test Set'],
    y=y[1],
    name='Yes',
    marker=dict(color='#ff3300'),
    opacity=0.7
)
data = [trace0, trace1]
layout = go.Layout(
    title='Count of classes in each set',
    xaxis={'title': 'Set'},
    yaxis={'title': 'Count'}
)
fig = go.Figure(data, layout)
fig.show()

def plot_samples(X, y, labels_dict, n=50):
    """
    Plots a grid of sample images from the dataset.
    
    Parameters:
    - X: Array of images.
    - y: Array of labels corresponding to the images.
    - labels_dict: Dictionary mapping label indices to class names.
    - n: Number of images to display for each label.
    """
    for index in np.unique(y):  # Loop through unique labels in y
        # Ensure the index does not exceed the number of available samples
        if index >= len(X):
            print(f"Warning: Label index {index} exceeds the available sample size.")
            continue  # Loop through unique labels in y
        imgs = X[np.argwhere(y == index)][:n]  # Select images for the label
        j = 10
        i = int(n / j)

        plt.figure(figsize=(15, 6))
        c = 1
        for img in imgs:
            plt.subplot(i, j, c)
            img_rgb = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)  # Convert to RGB
            plt.imshow(img_rgb / 255.0)

            plt.xticks([])
            plt.yticks([])
            c += 1
        plt.suptitle('Tumor: {}'.format(labels_dict[index]))
        plt.show()

# Plot sample images from the training set
plot_samples(X_train, y_train, labels, 30)

def crop_imgs(set_name, target_size=(224, 224)):
    """
    Crops the region of interest from each image in the dataset and resizes to a fixed size.
    """
    cropped_imgs = []
    for img in set_name:
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply a binary threshold
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # Convert to uint8 explicitly (important for findContours)
        thresh = thresh.astype(np.uint8)
        
        # Find contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        # If no contours are found, skip this image
        if len(cnts) == 0:
            continue
        
        # Get the largest contour (assumes the largest contour is the object of interest)
        c = max(cnts, key=cv2.contourArea)
        
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(c)
        
        # Crop the image to the bounding box
        cropped = img[y:y+h, x:x+w]
        
        # Resize the cropped image to the target size
        resized_cropped = cv2.resize(cropped, target_size)
        cropped_imgs.append(resized_cropped)
    
    return np.array(cropped_imgs)

X_train_crop = crop_imgs(set_name=X_train)
X_val_crop = crop_imgs(set_name=X_val)
X_test_crop = crop_imgs(set_name=X_test)

# Plot cropped images
plot_samples(X_train_crop, y_train, labels, 30)

def save_new_images(x_set, y_set, folder_name):
    i = 0
    for (img, imclass) in zip(x_set, y_set):
        if imclass == 0:
            cv2.imwrite(folder_name + 'NO/' + str(i) + '.jpg', img)
        else:
            cv2.imwrite(folder_name + 'YES/' + str(i) + '.jpg', img)
        i += 1

save_new_images(X_train_crop, y_train, folder_name='C:/Users/AIGow/OneDrive/Desktop/Python/Brain/output/train')
save_new_images(X_val_crop, y_val, folder_name='C:/Users/AIGow/OneDrive/Desktop/Python/Brain/output/val')
save_new_images(X_test_crop, y_test, folder_name='C:/Users/AIGow/OneDrive/Desktop/Python/Brain/output/test')

def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-16 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_new)

# Preprocess the images correctly
X_train_prep = preprocess_imgs(set_name=X_train_crop, img_size=IMG_SIZE)  # Fix here, use X_train_crop
X_test_prep = preprocess_imgs(set_name=X_test_crop, img_size=IMG_SIZE)
X_val_prep = preprocess_imgs(set_name=X_val_crop, img_size=IMG_SIZE)

# Check data sizes before calling plot_samples
print(f"X_train_prep shape: {X_train_prep.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Unique labels in y_train: {np.unique(y_train)}")

# Plot samples from the preprocessed images
plot_samples(X_train_prep, y_train, labels, 30)

# CNN MODEL

demo_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1./255,
    shear_range=0.05,
    brightness_range=[0.1, 1.5],
    horizontal_flip=True,
    vertical_flip=True
)

plt.imshow(X_train_crop[0])
plt.xticks([])
plt.yticks([])
plt.title('Original Image')
plt.show()

plt.figure(figsize=(15, 6))
i = 1
if os.path.exists('preview'):
    plt.figure(figsize=(15, 6))
    i = 1
    for img_name in os.listdir('preview'):
        img_path = os.path.join('preview', img_name)
        
        if os.path.isfile(img_path):  # Ensure it's a file
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(3, 7, i)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            i += 1
            if i > 3 * 7:
                break
    plt.suptitle('Augmented Images')
    plt.show()
else:
    print("Directory 'preview' not found!")


train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    seed=42  # Random seed
)


validation_generator = test_datagen.flow_from_directory(
    VAL_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode='binary',
    seed=42  # Random seed
)

# Use pre-trained weights from ImageNet
base_model = VGG16(
    weights='imagenet',  # Use ImageNet weights
    include_top=False, 
    input_shape=IMG_SIZE + (3,)
)

NUM_CLASSES = 1

model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

# Freezing the convolutional base
model.layers[0].trainable = False

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

model.summary()

EPOCHS = 30
es = EarlyStopping(
    monitor='val_acc', 
    mode='max',
    patience=6
)

history = model.fit(
    train_generator,
    steps_per_epoch=50,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=25,
    callbacks=[es]
)

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(history.epoch) + 1)

plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Set')
plt.plot(epochs_range, val_acc, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Set')
plt.plot(epochs_range, val_loss, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')

plt.tight_layout()
plt.show()

# Save the model
model.save('brain_tumor_vgg16_model.h5')

