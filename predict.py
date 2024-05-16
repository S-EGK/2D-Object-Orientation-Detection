import tensorflow as tf
import cv2
import os
import numpy as np

# Load the trained UNet model
model = tf.keras.models.load_model("Model\\model_checkpoint_epoch_0500.h5")

# Define path to test data
test_dict = "test"

# Get the list of test images
test_file_names = os.listdir(test_dict)

# Load and preprocess the input image
input_image_path = f"test\\test15.jpg"
input_image = cv2.imread(input_image_path)
resized_image = cv2.resize(input_image, (256, 256))
resized_image = resized_image / 255.0  # Normalize the pixel values to [0, 1]

# Make prediction
predicted_mask = model.predict(np.expand_dims(resized_image, axis=0))[0]

# Apply threshold
threshold = 0.1
binary_mask = (predicted_mask > threshold).astype(np.uint8)

# Resize the binary mask to the original image size
resized_binary_mask = cv2.resize(binary_mask, (input_image.shape[1], input_image.shape[0]))

# Save the output image
output_image_path = f"test_mask\\test15.png"
cv2.imwrite(output_image_path, resized_binary_mask * 255)  # Rescale values to [0, 255] for saving