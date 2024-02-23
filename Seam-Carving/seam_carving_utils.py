import scipy
import numpy as np
from scipy.ndimage.filters import convolve


def get_energy_pixel_array(image_array):
    """
    Calculates the energy array via the Sobel filter
    """
    return scipy.ndimage.sobel(image_array)


def calc_energy(pixel_array):
    filter_du = np.array(
        [
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
        ]
    )
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array(
        [
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [1.0, 0.0, -1.0],
        ]
    )
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dv = np.stack([filter_dv] * pixel_array.shape[-1], axis=2)

    pixel_array = pixel_array.astype("float32")
    convolved = np.absolute(convolve(pixel_array, filter_du)) + np.absolute(
        convolve(pixel_array, filter_dv)
    )

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)

    return energy_map


def get_pixel_array_for_imgmob(unmapped_array):
    return (unmapped_array / np.max(unmapped_array)) * 255


def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index -1
            if j == 0:
                idx = np.argmin(M[i - 1, j : j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1 : j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack


def carve_column(img):
    r, c, num_channels = img.shape

    M, backtrack = minimum_seam(img)

    # Create a (r, c) matrix filled with the value True
    # We'll be removing all pixels from the image which
    # have False later
    mask = np.ones((r, c), dtype=np.bool)

    # Find the position of the smallest element in the
    # last row of M
    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        # Mark the pixels for deletion
        mask[i, j] = False
        j = backtrack[i, j]

    # Since the image has 3 channels, we convert our
    # mask to 3D
    mask = np.stack([mask] * num_channels, axis=2)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    img = img[mask].reshape((r, c - 1, num_channels))

    return img


def crop_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in range(c - new_c):  # use range if you don't want to use tqdm
        print(f"Carving progress: {i / (c - new_c)}")
        img = carve_column(img)

    return img


def generate_all_seams_recursively(pixel_array, top_pixel=2):
    """
    Function to generate all seams recursively. Given an input top pixel and
    the actual array, we traverse the array recursively and obtain all the possible
    seams under that particular top pixel.

    Ouput is in [[(row, col), ...], ...] format. That is, an array of sorted arrays
    of coordinates that list the steps to follow for each seam.

    Given the number of rows, n, this will output less than 3^n possible seams. It's
    a bit less since those seams that are out of bounds won't be returned.
    """
    rows, cols = pixel_array.shape
    seams_array = []

    def traverse(arr, seam: list, r, c):
        # If the current position is the bottom-right corner of
        # the matrix
        if r >= rows - 1:

            seam.append((r, c))
            seams_array.append(seam[: r + 1])
            seam.pop()

            return

        # Print the value at the current position
        seam.append((r, c))

        # If the end of the current row has not been reached
        if c - 1 >= 0:
            traverse(arr, seam, r + 1, c - 1)

        traverse(arr, seam, r + 1, c)

        if c + 1 < cols:
            traverse(arr, seam, r + 1, c + 1)

        seam.pop()

    traverse(pixel_array, [], 0, top_pixel)
    return seams_array
