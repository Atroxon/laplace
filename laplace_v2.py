import cv2 as cv
import matplotlib.pyplot as plt 
import numpy as np
import copy

test_files = ['images/3.jpeg']

AVG_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
L_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
E_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])


def main():
    open_imgs = import_images(paths=test_files)

    derivatives = list()
    # First kernel
    for img in open_imgs:
        average_matrix = convolution_2D(img, AVG_kernel)
        laplace_matrix = convolution_2D(average_matrix, L_kernel)
        absolute_matrix = absolute(laplace_matrix)
        normalized_matrix = min_max(absolute_matrix,(0,255))
        derivatives.append(normalized_matrix)

    # for i in range(len(open_imgs)):
    #     plot_derivatives(open_imgs[i],derivatives[i])
    
    # Second kernel
    for img in open_imgs:
        average_matrix = convolution_2D(img, AVG_kernel)
        laplace_matrix = convolution_2D(average_matrix, E_kernel)
        absolute_matrix = absolute(laplace_matrix)
        normalized_matrix = min_max(absolute_matrix,(0,255))
        derivatives.append(normalized_matrix)

    plot_derivatives(open_imgs[0],derivatives[0])
    plot_derivatives2(open_imgs[0],derivatives[1])
    
    plt.show()

def import_images(paths=None):
    """
    Import images from input paths
    inputs:
        paths: list containing paths to images
    output:
        imgs: list containing cv2 objects
    """
    imgs = list()
    if paths == None:
        out = dummy()
        imgs.append(out)
        test = copy.deepcopy(imgs)
        return test
        
    for path in paths:
        imgs.append(cv.imread(path, cv.IMREAD_GRAYSCALE))
    return imgs


def convolution_2D(image, kernel, padding=0, strides=1): #Add strides
    print(image)
    xImage = image.shape[0]
    yImage = image.shape[1]
    xKernel = kernel.shape[0]
    yKernel = kernel.shape[1]

    # Add padding
    pad_image = np.zeros((xImage+(padding*2),yImage+(padding*2)), np.int16)
    pad_image[padding:(xImage)+padding, padding:(yImage)+padding] = image
    print(pad_image)
    print(pad_image.shape)

    # Adjusted to the spaces where kernel "fits"
    out_matrix = np.zeros((pad_image.shape[0]-2*(xKernel//2),pad_image.shape[1]-2*(yKernel//2)), np.int16)
    print(out_matrix.shape)

    for x in range(pad_image.shape[0]): #Add strides here
        abs_x = x - (xKernel//2)
        if x < (xKernel//2): #Upper bounds
            continue
        if x >= pad_image.shape[0]-(xKernel//2): #Lower bounds
            break

        for y in range(pad_image.shape[1]):
            abs_y = y - (yKernel//2)
            if y < (yKernel//2): #Left bound
                continue
            if y >= pad_image.shape[1]-(yKernel//2): #Right bound
                break
            
            extract = pad_image[x-(xKernel//2):x+(xKernel//2)+1, y-(yKernel//2):y+(yKernel//2)+1]
            pixel_val = np.sum([kernel * extract])
            out_matrix[abs_x, abs_y] = pixel_val

    return out_matrix


def plot_derivatives(image, matrix):
    popFig,popAxs=plt.subplots(1,2)
    popFig.suptitle("Grayscale ")
    
    popAxs[0].imshow(image, cmap='gray')
    popAxs[1].imshow(matrix, cmap='gray')
    return

def plot_derivatives2(image, matrix):
    ppFig,ppAxs=plt.subplots(1,2)
    ppFig.suptitle("Grayscale ")
    
    ppAxs[0].imshow(image, cmap='gray')
    ppAxs[1].imshow(matrix, cmap='gray')
    return


def absolute(matrix):
    absolute_matrix = np.zeros((matrix.shape[0],matrix.shape[1]), np.int16)
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            absolute_matrix[x,y] = abs(matrix[x,y])    
    return absolute_matrix


def min_max(matrix, boundaries):
    LBound, UBound = boundaries
    m = matrix.min()
    M = matrix.max()

    normalized = np.zeros((matrix.shape[0],matrix.shape[1]), np.uint8)

    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
  
            normalized[x,y] = (matrix[x,y]-LBound)*(UBound-LBound)/(M-m) # todo eso +LBound
    print(normalized)
    return normalized



if __name__ == '__main__':
    main()