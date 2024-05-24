import cv2
import numpy as np

def calculate_rotation_angle(template_mask, test_mask):
    # Find contours in the template mask
    template_contours, _ = cv2.findContours(template_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find contours in the test mask
    test_contours, _ = cv2.findContours(test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if both images have at least one object
    if len(template_contours) == 0 or len(test_contours) == 0:
        print("Error: Both images should have at least one object.")
        return None
    
    # Fit rotated bounding box around the object in the template mask
    template_rect = cv2.minAreaRect(template_contours[0])
    
    # Find the largest object in the test mask
    largest_area = 0
    largest_contour = None
    for contour in test_contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour
    
    # Check if a contour was found in the test mask
    if largest_contour is None:
        print("Error: Unable to find a contour in the test mask.")
        return None, None
    
    # Fit rotated bounding box around the largest object in the test mask
    test_rect = cv2.minAreaRect(largest_contour)

    # Calculate the orientation of the object in the template mask
    template_orientation = cv2.fitEllipse(template_contours[0])[2]
    
    # Calculate the orientation of the object in the test mask
    test_orientation = cv2.fitEllipse(test_contours[0])[2]
    
    # Calculate rotation angle of the test object relative to the template object
    rotation_angle = test_orientation - template_orientation

    # Calculate rotation angle of the test object bbox relative to the template object bbox
    bbox_rotation_angle = test_rect[2] - template_rect[2]
    
    return rotation_angle, bbox_rotation_angle, test_rect

def find_rotation(template_img, test_img):
    # Resize template image to quarter of its original size
    template_resized = cv2.resize(template_img, None, fx=0.25, fy=0.25)

    # Resize test image to match the size of the resized template image
    test_resized = cv2.resize(test_img, (template_resized.shape[1], template_resized.shape[0]))

    # Convert images to grayscale
    template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_resized, cv2.COLOR_BGR2GRAY)

    # Use ORB detector to find keypoints and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(test_gray, None)

    # Match descriptors between the template and test images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate affine transformation
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    # Calculate rotation angle
    angle_rad = np.arctan2(M[1, 0], M[0, 0])
    angle_deg = np.degrees(angle_rad)

    return angle_deg

# Load template image and mask
template_image = cv2.imread("Template\\template1.jpg")
template_mask = cv2.imread("Mask\\template1.png", cv2.IMREAD_GRAYSCALE)

# Load test image and mask
test_image = cv2.imread("test\\test4.jpg")
test_mask = cv2.imread("test_mask\\test4.png", cv2.IMREAD_GRAYSCALE)

# Calculate rotation angle and test rectangle from mask
rotation_angle, bbox_rotation_angle, test_rect = calculate_rotation_angle(template_mask, test_mask)

# Calculate rotation angle from images
angle_deg = find_rotation(template_image, test_image)

# Draw bounding box around the object in the test image
test_image_with_bbox = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
box = cv2.boxPoints(test_rect)
box = np.intp(box)
cv2.drawContours(test_image_with_bbox, [box], 0, (0, 255, 0), 2)

# Resize the images and masks to 25% of their size before displaying
resized_test_image_with_bbox = cv2.resize(test_image_with_bbox, None, fx=0.25, fy=0.25)
resized_template_image = cv2.resize(template_image, None, fx=0.25, fy=0.25)
resized_template_mask = cv2.resize(template_mask, (resized_template_image.shape[1], resized_template_image.shape[0]))
resized_test_mask = cv2.resize(test_mask, (resized_test_image_with_bbox.shape[1], resized_test_image_with_bbox.shape[0]))

# Convert grayscale masks to color (3-channel) images
resized_template_mask_color = cv2.cvtColor(resized_template_mask, cv2.COLOR_GRAY2BGR)
resized_test_mask_color = cv2.cvtColor(resized_test_mask, cv2.COLOR_GRAY2BGR)

# Concatenate the images and their masks vertically
template_pair = cv2.vconcat([resized_template_image, resized_template_mask_color])
test_pair = cv2.vconcat([resized_test_image_with_bbox, resized_test_mask_color])

# Concatenate the template and test pairs horizontally
final_image = cv2.hconcat([template_pair, test_pair])

# Add rotation angle text to the final image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(final_image, 'Rotation Angle from mask: {:.2f} degrees'.format(rotation_angle), (10+300, 30), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
cv2.putText(final_image, 'Rotation Angle from image: {:.2f} degrees'.format(angle_deg), (10+300, 60), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
cv2.putText(final_image, 'Rotation Angle from bbox: {:.2f} degrees'.format(bbox_rotation_angle), (10+300, 90), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

# Display the final image
cv2.imshow("Final Template(left) and Test(right) Image", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the final image
cv2.imwrite("output\\final_image1_4.png", final_image)

# Print rotation angle
print("Rotation Angle from mask:", rotation_angle)
print("Rotation Angle from image:", angle_deg)
print("Rotation Angle from bbox:", bbox_rotation_angle)