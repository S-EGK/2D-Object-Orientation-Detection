#%% Import Libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator # type: ignore
import keras.backend as backend
import matplotlib.pyplot as plt
# from focal_loss import BinaryFocalLoss    #pip install focal-loss
import pickle

# Setup GPU for training
# Limit GPU usage
MEMORY_FRACTION = 0.8
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

# Params

# Define the paths to your dataset
image_dict = "Template"
mask_dict = "Mask"

# Get the list of image file names and upsample accordingly
# Training Set
image_file_names = os.listdir(image_dict)
mask_image_file_names = os.listdir(mask_dict)

# Repeat each image five times
image_file_names_repeated = [item for sublist in [[name]*5 for name in image_file_names] for item in sublist]
mask_image_file_names_repeated = [item for sublist in [[name]*5 for name in mask_image_file_names] for item in sublist]

# Training Set
train_file_names = image_file_names_repeated
mask_file_names = mask_image_file_names_repeated

# Define the batch size and image dimensions
image_height, image_width = 256, 256
num_channels = 3  # Assuming RGB images
batch_size = 8
num_epochs = 500
image_input_shape = (image_width, image_height, num_channels)
num_classes = 1     # binary classification

# Calculate the number of steps per epoch
num_train_samples = 0
num_train_samples += len(train_file_names)
steps_per_epoch = num_train_samples // batch_size

# Save model
save_model_every = 10
save_model_freq = int(save_model_every * steps_per_epoch)

# Folder path to save model checkpoints, metrics
save_path = "Model3"

if not os.path.exists(save_path):
    os.makedirs(save_path)


# UNet Model

def conv_block(inputs, filters, kernel_size):
    x = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same', kernel_initializer='HeNormal')(inputs)
    x = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same', kernel_initializer='HeNormal')(x)

    return x

def unet(input_shape, num_classes):

    # Input Layer
    inputs = tf.keras.Input(shape=input_shape)

    # Contracting Path (Encoder)
    conv1 = conv_block(inputs, 64, 3)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128, 3)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 256, 3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 512, 3)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottom
    conv5 = conv_block(pool4, 1024, 3)

    # Expanding Path (Decoder)
    up6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    concat6 = tf.keras.layers.concatenate([conv4, up6], axis=3)
    conv6 = conv_block(concat6, 512, 3)

    up7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    concat7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = conv_block(concat7, 256, 3)

    up8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    concat8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = conv_block(concat8, 128, 3)

    up9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    concat9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = conv_block(concat9, 64, 3)

    # Output layer
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


#%% Data Generator

# Define a function to load, preprocess and augment the image and mask
def load_and_preprocess_image(image_path, mask_path):
    # Load the image and mask
    image = load_img(image_path, target_size=(image_height, image_width))
    mask = load_img(mask_path, target_size=(image_height, image_width), color_mode='grayscale')

    # Convert the image and mask to arrays
    image = img_to_array(image) / 255.0
    mask = img_to_array(mask) / 255.0

    # # Resize image and mask to size 256x256
    # # Calculate the maximum possible shift for cropping
    # image = tf.image.resize(image, (image_height, image_width))
    # mask = tf.image.resize(mask, (image_height, image_width))

    # Randomly rotate the image between the range of -90 to 90 to image and its mask
    if np.random.uniform() > 0.5:
        seed = np.random.randint(0, 1000)
        datagen_rotation = ImageDataGenerator(rotation_range=90)
        image = datagen_rotation.random_transform(image, seed=seed)
        mask = datagen_rotation.random_transform(mask, seed=seed)

    # Randomly apply brightness and contrast augmentation to the image only
    if np.random.uniform() > 0.5:
        image = tf.image.random_brightness(image, 0.2, seed=None)
        image = tf.image.random_contrast(image, 0.8, 1.2, seed=None)

    return image, mask

# Define a generator function to yield batches of data
def data_generator(train_file_names, mask_file_names, batch_size, image_dict, mask_dict):
    num_samples = len(train_file_names)
    while True:
        indices = np.random.choice(num_samples, batch_size, replace=False)
        batch_images = []
        batch_masks = []
        for index in indices:
            image_path = os.path.join(image_dict, train_file_names[index])
            mask_path = os.path.join(mask_dict, mask_file_names[index])
            image, mask = load_and_preprocess_image(image_path, mask_path)
            batch_images.append(image)
            batch_masks.append(mask)

        batch_images = np.array(batch_images)
        batch_masks = np.array(batch_masks)
        yield batch_images, batch_masks

#%% Compile and Train the model

# Define a callback to save the model at certain intervals
filename_template = "model_checkpoint_epoch_{epoch:04d}.h5"
checkpoint_filepath = os.path.join(save_path, filename_template)
save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                   save_weights_only=False,
                                                   save_freq=save_model_freq,
                                                   verbose=1)

# Create an instance of the UNet model and compile the model
model = unet(image_input_shape, num_classes)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy',
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall()])

# Create instances of the data generators
train_data_generator = data_generator(train_file_names, mask_file_names, batch_size, image_dict, mask_dict)

# Train the model using the batched data generators
history = model.fit(train_data_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=num_epochs,
                    callbacks=[save_callback])


#%% Plot and save the metrics

# Plot the accuracy and loss metrics during training
plt.figure()

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train'], loc='best')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train'], loc='best')
plt.grid()

plt.tight_layout()
plt.show()

plt.savefig(f"{save_path}/Accuracy_Loss.png")

# Plot the precision and recall metrics during training
plt.figure()

plt.subplot(1, 2, 1)
plt.plot(history.history['precision'])
plt.title('Model precision')
plt.xlabel('Epoch')
plt.ylabel('precision')
plt.legend(['Train'], loc='best')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['recall'])
plt.title('Model recall')
plt.xlabel('Epoch')
plt.ylabel('recall')
plt.legend(['Train'], loc='best')
plt.grid()

plt.tight_layout()
plt.show()

plt.savefig(f"{save_path}/Precision_Recall.png")


#%% Save the history (metrics)
history_filename = "history.pkl"
history_filepath = os.path.join(save_path, history_filename)
with open(history_filepath, 'wb') as file:
    pickle.dump(history.history, file)