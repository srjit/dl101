import numpy as np
import cv2

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"



horizontal_flip = True


class Imgdata():

    def __init__(self, class_list):
        self.instances, self.labels = self.create_train_data(class_list)        
        self.length = len(self.instances)
        self.pointer = 0
        self.num_classes = len(set(self.labels))

        self.mean = np.array([104., 117., 124.]) ## mean of images in CIFAR-10
        self.scaling_dim = [227, 227]


    def create_train_data(self, filepath):

        with open(filepath) as f:
            content = f.readlines()

        self.input_lines = [x.strip() for x in content] 

        instances = []
        labels = []

        def splitline(line):
            line = line.split(" ")
            instances.append(line[0])
            labels.append(line[1])
        
        list(map(splitline, self.input_lines))
        return instances, labels


    def get_batch(self, batch_size):
        
        instances_tosend = self.instances[self.pointer:self.pointer+batch_size]
        labels_tosend = self.labels[self.pointer:self.pointer+batch_size]
        labels_tosend = [int(x) for x in labels_tosend]

        self.pointer += batch_size

        ## reading image data
        for index, image in enumerate(instances_tosend):
            img = cv2.imread(image)
            
            img = cv2.resize(img, (self.scaling_dim[0], self.scaling_dim[1]))
            img = img.astype(np.float32)
            
            img -= self.mean
            instances_tosend[index] = img

        one_hot_labels = np.zeros((len(instances_tosend), self.num_classes))
        one_hot_labels[np.arange(len(instances_tosend)), labels_tosend] = 1        

        return instances_tosend, one_hot_labels

            
