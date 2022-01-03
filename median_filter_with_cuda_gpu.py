# Import important libraries
# Numpy is a library for performing calculations on matrices
import numpy as np
# Numba is a library for calculations that ultilize the CUDA programming
from numba import cuda
import numba
# Basic mathematical & time libraries
import math
import time
# OpenCV library for image processing
import cv2
# Matplotlib library for plotting the result
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------------------------------------------
# Test image path. Put this image at the same directory as the script file.
image_path = 'MinhDuc_testphoto.jpg'
# Number of iterations (or number of test time)
iterations = 10
# Median filter params
kernel_size = 11
win_flat_size = kernel_size * kernel_size
rn = math.floor(kernel_size/2)
med = math.ceil(win_flat_size/2)
# Image resize parameters
image_dimension = 1500
image_size = (image_dimension, image_dimension)


# Matrix mul params
row = col = 100


# Define matrix multiplication using GPU (CUDA)
@cuda.jit
def matrix_mul_gpu(A, B, C):
    # Perform matrix multiplication of C = A * B
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.5
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


# Define image median filter using GPU (CUDA)
@cuda.jit
def median_filter_gpu(img, out_img):

    row, col = cuda.grid(2)

    windows = cuda.local.array(win_flat_size, numba.int32)

    if row < out_img.shape[0] and col < out_img.shape[1]:
        count = 0
        for i in range(row - rn, row + rn + 1):
            for j in range(col - rn, col + rn + 1):
                if 0 <= i < out_img.shape[0] and 0 <= j < out_img.shape[1]:
                    windows[count] = img[i, j]
                else:
                    windows[count] = 0
                count += 1

        for i in range(win_flat_size):
            for j in range(win_flat_size):
                if windows[i] > windows[j]:
                    windows[i], windows[j] = windows[j], windows[i]

        out_img[row, col] = windows[med]


# Define test matrix multiplication using GPU (CUDA)
def test_matmul(row, col):
    
    host_data_1 = np.random.rand(row, col)
    host_data_2 = np.random.rand(row, col)

    device_data_1 = cuda.to_device(host_data_1)
    device_data_2 = cuda.to_device(host_data_2)

    out_device_data = cuda.device_array((row, col))
    time_cpu = []
    time_gpu = []

    for i in range(iterations):
        start = time.time()
        out = np.matmul(host_data_1, host_data_2)
        end = time.time()
        print("=====Interation", (i+1))
        print(f'====CPU time:{end - start} with shape {row, col}')
        if i != 0:
            time_cpu.append(end - start)

        threads_per_block = (16, 16)
        blocks_per_grid_x = int(math.ceil(out_device_data.shape[0] / threads_per_block[0]))
        blocks_per_grid_y = int(math.ceil(out_device_data.shape[1] / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        start = time.time()
        matrix_mul_gpu[blocks_per_grid, threads_per_block](device_data_1, device_data_2, out_device_data)
        end = time.time()

        print(f'====GPU time:{end - start} with shape {row, col}')

        if i != 0:
            time_gpu.append(end - start)

        out_gpu = out_device_data.copy_to_host()
        print(f'====Error with cpu compute:{np.sum(out - out_gpu)}')
        print()
        del out_gpu

    print(f"=====Average time CPU:{sum(time_cpu) / (iterations - 1)}")
    print(f"=====Average time GPU:{sum(time_gpu) / (iterations - 1)}")


def test_median_filter(img, img_size):

    img = cv2.resize(img, img_size)
    device_img = cuda.to_device(img)
    out_device_img = cuda.device_array(img.shape)

    time_cpu = []
    time_gpu = []
    for i in range(iterations):
        start = time.time()
        cpu_med = cv2.medianBlur(img, kernel_size)
        end = time.time()
        print("----Inter ",(i+1))
        print(f'====CPU time:{end - start} with image shape {img_size}')
        if i != 0:
            time_cpu.append(end - start)

        threads_per_block = (16, 16)
        blocks_per_grid_x = int(math.ceil(img.shape[0] / threads_per_block[0]))
        blocks_per_grid_y = int(math.ceil(img.shape[1] / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        start = time.time()
        median_filter_gpu[blocks_per_grid, threads_per_block](device_img, out_device_img)
        end = time.time()
        print(f'====GPU time:{end - start} with img shape {img_size}')

        if i != 0:
            time_gpu.append(end - start)

        out_gpu_img = out_device_img.copy_to_host()

    # Draw image results after applying median filter, compare CPU results to GPU results to original.
    plt.figure(figsize=(9, 3))

    ax1 = plt.subplot(131)
    plt.imshow(img, cmap='gray')
    ax1.title.set_text('Original image')

    ax2 = plt.subplot(132)
    plt.imshow(cpu_med, cmap='gray')
    ax2.title.set_text('CPU median')

    ax3 = plt.subplot(133)
    plt.imshow(out_gpu_img.astype(int), cmap='gray')
    ax3.title.set_text('GPU median')

    plt.suptitle(f'Median filter image size {img_size}, kernel size {kernel_size}')
    plt.show()
    print("**********************")
    print(f"=====Average time CPU:{sum(time_cpu) / (iterations - 1)}")
    print(f"=====Average time GPU:{sum(time_gpu) / (iterations - 1)}")
    print()


# Main function
if __name__ == '__main__':
    print('--------Test median filter--------')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    test_median_filter(image, image_size)
    print("-----------------------------")