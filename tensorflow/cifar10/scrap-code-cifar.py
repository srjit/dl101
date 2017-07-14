
import tests
import second

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"




random100 = tests.random_batch(100)
x_batch = second.pre_process_images(images_train[random100, :, :, :], istrain=True)

#distorted_images = pre_process_images(images_train, istrain=True)

