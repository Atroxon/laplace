import cv2
import matplotlib as plot
import numpy as np

test_files = ['images/byn.jpg']

L_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])

def main():
    open_imgs = import_images()

    derivatives = list()
    for img in open_imgs:
        der_matrix = laplace_filter_2D(img,test_k)
        # derivatives.append(der_matrix)

    # for i in range(len(open_imgs)):
    #     plot_derivatives(img[i],der_matrix[i])


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
        # img_1D = np.random.randint(0,255,(12))
        # imgs.append(img_1D)
        img_2D = np.random.randint(0,255,(12,12))
        imgs.append(img_2D)
        return imgs
        
    for path in paths:
        imgs.append(cv.resize(cv.imread(path),(400,300)))
    return imgs


def laplace_filter_2D(image, kernel, padding=1, strides=1): #Add strides

    xImage = image.shape[0]
    yImage = image.shape[1]
    xKernel = kernel.shape[0]
    yKernel = kernel.shape[1]

    # Add padding
    pad_image = np.zeros((xImage+(padding*2),yImage+(padding*2)), np.uint8)
    pad_image[padding:(xImage)+padding, padding:(yImage)+padding] = image
    print(pad_image)
    print(pad_image.shape)

    # Adjusted to the spaces where kernel "fits"
    out_matrix = np.zeros((pad_image.shape[0]-2*(xKernel//2),pad_image.shape[1]-2*(yKernel//2)), np.uint8)
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

    print(out_matrix)
    return out_matrix

def plot_derivatives(image, der_matrix):
    return





if __name__ == '__main__':
    main()