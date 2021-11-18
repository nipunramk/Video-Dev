from manim import *
import cv2
config["assets_dir"] = 'assets'

class ImageUtils(Scene):
    def construct(self):
        NUM_PIXELS = 32
        HEIGHT = 4
        new_image = self.down_sample_image("duck", NUM_PIXELS, NUM_PIXELS, image_height=HEIGHT)
        print('New down-sampled image shape:', new_image.get_pixel_array().shape)
        self.wait()
        self.add(new_image)
        self.wait()
        pixel_grid = self.get_pixel_grid(new_image, NUM_PIXELS)
        self.add(pixel_grid)
        self.wait()
        pixel_array = new_image.get_pixel_array().copy()

        pixel_array[-1, 6, :] = [0, 0, 255, 255]
        pixel_array[-2, 5, :] = [0, 0, 255, 255]
        pixel_array[-2, 6, :] = [0, 0, 255, 255]
        pixel_array[-1, 5, :] = [0, 0, 255, 255]

        
        adjusted_image = self.get_image_mob(pixel_array, height=HEIGHT)
        # self.remove(pixel_grid)
        # self.wait()
        self.play(
            new_image.animate.become(adjusted_image),
            # pixel_grid.animate.become(pixel_grid.copy())
        )
        self.wait()

    def get_image_mob(self, pixel_array, height=4):
        """
        @param pixel_array: multi-dimensional np.array[uint8]
        @return: ImageMobject of pixel array with given height
        """
        # print(pixel_array.shape, pixel_array)
        image = ImageMobject(pixel_array)
        image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        image.height = height
        return image

    def down_sample_image(self, filepath, num_horiz_pixels, num_vert_pixels, image_height=4):
        """
        @param: filepath - file name of image to down sample
        @param: num_horiz_pixels - number of horizontal pixels in down sampled image
        @param: num_vert_pixels - number of vertical pixels in down sampled image
        """
        assert num_horiz_pixels == num_vert_pixels, 'Non-square downsampling not supported'
        original_image = ImageMobject(filepath)
        original_image_pixel_array = original_image.get_pixel_array()
        width, height, num_channels = original_image_pixel_array.shape
        horizontal_slice = self.get_indices(width, num_horiz_pixels)
        vertical_slice = self.get_indices(height, num_vert_pixels)
        new_pixel_array = self.sample_pixel_array_from_slices(original_image_pixel_array, horizontal_slice, vertical_slice)
        new_image = ImageMobject(new_pixel_array)
        new_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        new_image.height = image_height
        assert new_pixel_array.shape[0] == num_horiz_pixels and new_pixel_array.shape[1] == num_vert_pixels, self.get_assert_error_message(new_pixel_array, num_horiz_pixels, num_vert_pixels)
        return new_image

    def sample_pixel_array_from_slices(self, original_pixel_array, horizontal_slice, vertical_slice):
        return original_pixel_array[horizontal_slice][:, vertical_slice]
     
    def get_indices(self, size, num_pixels):
        """
        @param: size of row or column
        @param: down-sampled number of pixels
        @return: an array of indices with len(array) = num_pixels
        """
        array = []
        index = 0
        float_index = 0
        while float_index < size:
            index = round(float_index)
            array.append(index)
            float_index += size / num_pixels
        return np.array(array)

    def get_assert_error_message(self, new_pixel_array, num_horiz_pixels, num_vert_pixels):
        return f'Resizing performed incorrectly: expected {num_horiz_pixels} x {num_vert_pixels} but got {new_pixel_array.shape[0]} x {new_pixel_array.shape[1]}'

    def get_pixel_grid(self, image, num_pixels_in_dimension):
        side_length_single_cell = image.height / num_pixels_in_dimension
        pixel_grid = VGroup(*[Square(side_length=side_length_single_cell).set_stroke(width=1) for _ in range(num_pixels_in_dimension ** 2)])
        pixel_grid.arrange_in_grid(rows=num_pixels_in_dimension, buff=0)
        return pixel_grid

    def get_yuv_image_from_rgb(self, pixel_array):

        # discard alpha channel
        rgb_img = pixel_array[:, :, :3]
        # channels need to be flipped to BGR for openCV processing
        rgb_img = rgb_img[:, :, [2, 1, 0]]
        img_yuv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)

        lut_u, lut_v = make_lut_u(), make_lut_v()

        # Convert back to BGR so we display the images
        y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
        u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
        v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

        u_mapped = cv2.LUT(u, lut_u)
        v_mapped = cv2.LUT(v, lut_v)

        # Flip channels to RGB
        y_rgb = y[:, :, [2, 1, 0]]
        u_mapped_rgb = u_mapped[:, :, [2, 1, 0]]
        v_mapped_rgb = v_mapped[:, :, [2, 1, 0]]


        return y_rgb, u_mapped_rgb, v_mapped_rgb


class TestGrayScaleImages(ImageUtils):
    def construct(self):
        pixel_array = np.uint8([[63, 0, 0, 0],
                                        [0, 127, 0, 0],
                                        [0, 0, 191, 0],
                                        [0, 0, 0, 255]
                                        ])
        image = self.get_image_mob(pixel_array, height=2)

 
        self.add(image)
        self.wait()

        new_pixel_array = image.get_pixel_array()
        print(new_pixel_array.shape)
        new_pixel_array[3][1] = 255
        new_image = self.get_image_mob(new_pixel_array, height=2)
        self.remove(image)
        self.wait()
        self.add(new_image)
        self.wait()
        next_image_pixel_array = new_image.get_pixel_array()
        next_image_pixel_array[1, 3, :] =  [255, 255, 255, 255]
        next_image = self.get_image_mob(next_image_pixel_array, height=2)
        self.remove(new_image)
        self.wait()
        self.add(next_image)
        self.wait()

class TestColorImage(ImageUtils):
    def construct(self):
        NUM_PIXELS = 32
        HEIGHT = 4
        new_image = self.down_sample_image("duck", NUM_PIXELS, NUM_PIXELS, image_height=HEIGHT)

        next_image_pixel_array = new_image.get_pixel_array()
        next_image_pixel_array[-1, 0, :] = [255, 255, 255, 255]
        next_image = self.get_image_mob(next_image_pixel_array, height=HEIGHT)
        self.add(next_image)
        self.wait()

class TestYCbCrImages(ImageUtils):
    def construct(self):
        original_image = ImageMobject("shed")
        y, u, v = self.get_yuv_image_from_rgb(original_image.get_pixel_array())
        y_channel = ImageMobject(y)
        u_channel = ImageMobject(u)
        v_channel = ImageMobject(v)

        original_image.move_to(LEFT * 2)
        y_channel.move_to(RIGHT * 2 + UP * 2)
        u_channel.move_to(RIGHT * 2 + UP * 0)
        v_channel.move_to(RIGHT * 2 + DOWN * 2)

        self.add(original_image)
        self.wait()

        self.add(y_channel, u_channel, v_channel)
        self.wait()

class TestYCbCrImagesDuck(ImageUtils):
    def construct(self):
        NUM_PIXELS = 32
        HEIGHT = 2
        new_image = self.down_sample_image("duck", NUM_PIXELS, NUM_PIXELS, image_height=HEIGHT)
        y, u, v = self.get_yuv_image_from_rgb(new_image.get_pixel_array())
        y_channel = self.get_image_mob(y, height=HEIGHT)
        u_channel = self.get_image_mob(u, height=HEIGHT)
        v_channel = self.get_image_mob(v, height=HEIGHT)

        new_image.move_to(LEFT * 2)
        y_channel.move_to(RIGHT * 2 + UP * 2)
        u_channel.move_to(RIGHT * 2 + UP * 0)
        v_channel.move_to(RIGHT * 2 + DOWN * 2)

        self.add(new_image)
        self.wait()

        self.add(y_channel, u_channel, v_channel)
        self.wait()


def make_lut_u():
    return np.array([[[i, 255-i, 0] for i in range(256)]], dtype=np.uint8)

def make_lut_v():
    return np.array([[[0, 255-i, i] for i in range(256)]], dtype=np.uint8)


def dct1D(f, N):
    result = []
    constant = (2 / N) ** 0.5
    for u in range(N):
        component = 0
        if u == 0:
            factor = 1 / np.sqrt(2)
        else:
            factor = 1
        for i in range(N):
            component += (constant * factor * np.cos(np.pi * u / (2 * N) * (2 * i + 1)) * f(i))

        result.append(component)

    return result


def f(i):
    return np.cos((2 * i + 1) * 3 * np.pi / 16)

def plot_function(f, N):
    import matplotlib.pyplot as plt 

    x = np.arange(0, N, 0.001)
    y = f(x)

    plt.plot(x, y)
    plt.show()

def g(i):
    return np.cos((2 * i + 1) * 7 * np.pi / 16)

def h(i):
    return np.cos((2 * i + 1) * 3 * np.pi / 16) * np.cos((2 * i + 1) * 1.5 * np.pi / 16)

def get_dct_elem(i, j):
    return np.cos(j * (2 * i + 1) * np.pi / 16)

def get_dct_matrix():
    matrix = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            matrix[j][i] = get_dct_elem(i, j)

    return matrix

def get_dot_product_matrix():
    dct_matrix = get_dct_matrix()
    dot_product_matrix = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            value = np.dot(dct_matrix[i], dct_matrix[j])
            if np.isclose(value, 0):
                dot_product_matrix[i][j] = 0
            else:
                dot_product_matrix[i][j] = value 
            


    return dot_product_matrix
