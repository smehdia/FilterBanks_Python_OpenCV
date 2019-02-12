import numpy as np
import matplotlib.pyplot as plt

## REFERENCE:
## http://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html


def make_filter(kernel_size, sigma, tau):
    x, y = np.meshgrid(np.arange(-(kernel_size-1)/2, (kernel_size+1)/2, 1), np.arange(-(kernel_size-1)/2, (kernel_size+1)/2, 1))
    r = np.sqrt(np.square(x) + np.square(y))
    f = np.multiply(np.cos(r * (np.pi * tau / sigma)), np.exp(-(np.square(r)) / (2 * sigma * sigma)))
    f = f - np.mean(np.mean(f))
    f /= np.sum(np.sum(np.abs(f)))

    return f

def build_schmid_filterbank(kernel_size):
    filters = []
    for sigma, tau in [(2, 1), (4, 1), (4, 2), (6, 1), (6, 2), (6, 3), (8, 1), (8, 2), (8, 3), (10, 1), (10, 2), (10, 3), (10, 4)]:
        filters.append(make_filter(kernel_size, sigma, tau))
    filters = np.array(filters)
    return filters


if __name__ == "__main__":

    KSIZE = 7
    filters = build_schmid_filterbank(KSIZE)

    fig = plt.figure(figsize=(8, 8))
    columns = 6
    rows = 2
    for i in range(1, columns * rows):
        fig.add_subplot(rows, columns, i)
        plt.imshow(filters[i], cmap='gray')
        plt.axis('off')
    plt.savefig('Schmid_FilterBank.png')
    plt.show()
