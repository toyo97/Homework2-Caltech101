from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        """
        Load in the main memory all the images defined in the split file along with their class label.
        To match the class index to the actual label name, find the index in the label_names attribute of the object.

        :param root: path to root directory
        :param split: name of the txt file containing all the relative paths of the images in the split
        :param transform: torchvision transforms object, transformation applied on the image during the access
        :param target_transform: torchvision transforms object, transformation applied on the target
        """
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split  # This defines the split you are going to use
        # (split files are called 'train.txt' and 'test.txt')

        '''
        INSTRUCTIONS
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

        # All data in memory approach
        self.label_names = {}

        self.imgs = []
        self.labels = []

        class_counter = 0
        for line in open(split + '.txt', 'r'):
            if line.find('BACKGROUND') == -1:
                img = pil_loader(os.path.join(data_dir, line.strip()))
                self.imgs.append(img)

                # Extract label name and, if not already seen, associate an incremental id
                label_name = line.split('/')[0]

                if label_name not in self.label_names:
                    self.label_names[label_name] = class_counter
                    class_counter += 1

                self.labels.append(self.label_names[label_name])

        assert class_counter == 101

    def __getitem__(self, index) -> (Image, int):
        """
        __getitem__ should access an element through its index

            :param index: int, direct access index
            :return: tuple: (sample, target) where target is class_index of the target class.
        """

        image, label = self.imgs[index], self.labels[index]  # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components

        :return: int
        """
        length = len(self.imgs)  # Provide a way to get the length (number of elements) of the dataset
        return length


# Testing the methods
if __name__ == '__main__':

    data_dir = '101_ObjectCategories'
    cal = Caltech(data_dir, split='test')
    image, label = cal[0]
    image.show()
    print(len(cal))
