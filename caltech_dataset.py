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
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        split_fpath = os.path.join(root, '..', split + '.txt' )

        # Correspondence label int<->str
        label_str_to_int = {}
        label_int_to_str = []
        curr_index = 0

        # Dataset data
        imgs = []
        labels = []

        with open(split_fpath, 'r') as f:
            for line in f:
                rel_path_splitted = line.split('/')
                label_str, img_name = rel_path_splitted[0], rel_path_splitted[1]
                img_path = os.path.join(root, label_str, img_name)
                print(f"img_path: {img_path}")
                img = pil_loader(img_path)
                
                # Build a correspondence between integer value label and string label
                if label_str not in label_str_to_int:
                    label_str_to_int[label_str] = curr_index
                    label_int_to_str.append(label_str)
                    curr_index += 1

                label_int = label_str_to_int[label_str]
                
                imgs.append(img)
                labels.append(label_int)

        self.n_classes = curr_index

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.imgs[index], self.labels[index]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.imgs)
        return length
