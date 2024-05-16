# 2D Object Orientation Detection

## Objective
The goal of this project is to detect the orientation of an object in a 2D plane. Given template images of the object of interest, the program finds the object in the test images and determines the rotation angle between the object in the template image and the object in the test images.

## Project Structure
1. **Train a Segmentation Model**: A UNet model is trained to segment the object of interest from template images.
2. **Create Masks for Test Images**: The trained UNet model is used to create masks for the test images.
3. **Calculate Rotation Angle**: The rotation angle between the object in the template and test images is calculated, and a bounding box is drawn around the object in the test image.

## Folders
- `Template`: Contains template images of the object of interest.
- `Mask`: Contains mask images corresponding to the template images.
- `test`: Contains test images where the object of interest needs to be detected.
- `test_mask`: Contains generated masks for the test images.
- `Model`: Contains the trained model checkpoints and metrics.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Pickle

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_link>
   cd <repository_name>
   ```
   
## Usage

### Step 1: Train the Segmentation Model
Run the following script to train the UNet model:
  ```bash
  python3 unet.py
  ```

This script will:
1. Load and preprocess the template images and their corresponding masks using an image generator.
2. Train a UNet model to segment the object of interest.
3. Save the trained model and training metrics in the 'Model' folder.

### Step 2: Create Masks for Test Images
Run the following script to generate masks for the test images:
```bash
  python3 predict.py
```

This script will:
1. Load the trained UNet model.
2. Generate masks for the test images.
3. Save the generated masks in the 'test_mask' folder.

### Step 3: Calculate Rotation Angle
Run the following script to calculate the rotation angle and draw bounding boxes:
```bash
  python3 main.py
```

This script will:
1. Load the template image, template mask, test image, and generated test mask.
2. Calculate the rotation angle between the object in the template image and the test image using the masks and images.
3. Draw a bounding box around the object in the test image.
4. Display the final image with bounding boxes and rotation angle.

## Example
Here is an example of the output image with the bounding box around the detected object and the rotation angle annotated.


## Acknowledgments
1. The UNet model architecture is inspired by the paper: U-Net: Convolutional Networks for Biomedical Image Segmentation.
2. Thanks to the contributors of the TensorFlow and Keras libraries for providing the tools needed to build and train the model.
