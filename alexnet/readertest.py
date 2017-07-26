from reader import Imgdata

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


filename = "train.txt"

train_handle = Imgdata(filename)

print(train_handle.pointer)
instances, labels = train_handle.get_batch(10)
