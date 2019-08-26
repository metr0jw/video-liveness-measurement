import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim


m12 = list()
m23 = list()
m13 = list()
s12 = list()
s23 = list()
s13 = list()
cpt = sum([len(files) for r, d, files in os.walk("ITZY")])
print(cpt)


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_imagesA(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    m12.append(m)
    s12.append(s)

    '''
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    # show the images
    # plt.show()
    '''


def compare_imagesB(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    m23.append(m)
    s23.append(s)

    '''
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    # show the images
    # plt.show()
    '''

def compare_imagesC(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    m13.append(m)
    s13.append(s)

    '''
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    # show the images
    # plt.show()
    '''


for file_count in range(0, cpt - 3):
    # load the images
    first = cv2.imread("ITZY/itzy_image" + str(file_count) + ".png")
    second = cv2.imread("ITZY/itzy_image" + str(file_count + 1) + ".png")
    third = cv2.imread("ITZY/itzy_image" + str(file_count + 2) + ".png")

    # convert the images to grayscale
    first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    third = cv2.cvtColor(third, cv2.COLOR_BGR2GRAY)
    # initialize the figure
    fig = plt.figure("Images")
    images = ("First", first), ("Second", second), ("Third", third)

    '''
    # loop over the images
    for (i, (name, image)) in enumerate(images):
        # show the image
        ax = fig.add_subplot(1, 3, i + 1)
        ax.set_title(name)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.axis("off")

    # show the figure
    plt.show()
    '''
    # compare the images
    compare_imagesA(first, second, "First vs. Second")
    compare_imagesB(second, third, "Second vs. Third")
    compare_imagesC(first, third, "First vs. Third")

df_m12 = pd.DataFrame(m12)
df_s12 = pd.DataFrame(s12)
df_m23 = pd.DataFrame(m23)
df_s23 = pd.DataFrame(s23)
df_m13 = pd.DataFrame(m13)
df_s13 = pd.DataFrame(s13)
df_concat = pd.concat([df_m12, df_m23, df_m13, df_s12, df_s23, df_s13], axis=1)

df_concat.to_csv("data_itzy.csv", mode='w', header=False)
print("DONE")