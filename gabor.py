import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_gabors(k_size):
    filters = []
    num_angles = 10

    for frequency in [5 / (2. * 2 * np.pi), 3 / (2. * 2 * np.pi), 1 / (2. * 2 * np.pi), 1 / (4. * 2 * np.pi),
                      1 / (8. * 2 * np.pi)]:
        sigma = 1 / frequency
        flag_orientation_0 = False
        for angle in range(num_angles):
            if (flag_orientation_0) and (angle == 0):
                continue
            if angle == 0:
                flag_orientation_0 = True
            g_kernel = cv2.getGaborKernel((k_size, k_size), sigma=float(sigma), theta=angle * np.pi / num_angles,
                                          lambd=1 / float(frequency),
                                          gamma=1., psi=0, ktype=cv2.CV_32F)
            filters.append(g_kernel)

    filters = np.array(filters)

    return filters

if __name__ == "__main__":

    KSIZE = 9
    filters = get_gabors(KSIZE)
    sqrt_num_filters = int(np.ceil(np.sqrt(filters.shape[0])))
    for i in range(sqrt_num_filters):
        for j in range(sqrt_num_filters):
            plt.subplot(sqrt_num_filters, sqrt_num_filters, j + sqrt_num_filters * i + 1)
            try:
                plt.imshow(filters[j + sqrt_num_filters * i, :, :], cmap='gray',
                           interpolation=None)
                plt.axis('off')
            except:
                continue
    plt.savefig('filterBank_Gabor.png')
    plt.show()
