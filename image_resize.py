import cv2

image = cv2.imread('MinhDuc_testphoto.jpg')
resize = cv2.resize(image, (500, 500))
cv2.imwrite('resize.jpg', resize)