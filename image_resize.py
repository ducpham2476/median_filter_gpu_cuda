# Import important libraries
# Image processing library: OpenCV
import cv2
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
# Put your desired image path here. Putting the image in the same directory as this script file is recommended
image_path = 'MinhDuc_testphoto.jpg'
image = cv2.imread(image_path)

width = 1000 # Or modify to your desired value
height = 1000 # Or modify to your desired value
desired_image_size = (width, height)

# Resize image
resize = cv2.resize(image, desired_image_size)

# Write image with new size
output_path = 'resize.jpg'
cv2.imwrite(output_path, resize)
