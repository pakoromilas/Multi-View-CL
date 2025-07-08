import numpy as np
import os
from glob import glob
from PIL import Image
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2

class AnimalDataset(data.dataset.Dataset):
  def __init__(self, base_dir, classes_file, disjoint_classes=False, train=True, attributes=False):
    predicate_binary_mat = np.array(np.genfromtxt(os.path.join(base_dir, 'predicate-matrix-binary.txt'), dtype='int'))
    self.predicate_binary_mat = predicate_binary_mat
    self.attributes = attributes

    if train:
        self.transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.Resize((224,224)), # ImageNet standard
        transforms.ToTensor()
        ])
    else:
        self.transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
        ])
    
    class_to_index = dict()
    # Build dictionary of indices to classes
    with open(os.path.join(base_dir, 'classes.txt')) as f:
      index = 0
      for line in f:
        class_name = line.split('\t')[1].strip()
        class_to_index[class_name] = index
        index += 1
    self.class_to_index = class_to_index

    img_names = []
    img_index = []
    with open(os.path.join(base_dir, '{}'.format(classes_file))) as f:
      for line in f:
        class_name = line.strip()
        FOLDER_DIR = os.path.join(os.path.join(base_dir, 'JPEGImages'), class_name)
        file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
        files = glob(file_descriptor)
        
        if disjoint_classes:
            if train:
                files = files[:int(0.8 * len(files))]
            else:
                files = files[int(0.8 * len(files)):]

        class_index = class_to_index[class_name]
        for file_name in files:
          img_names.append(file_name)
          img_index.append(class_index)
    self.img_names = img_names
    self.img_index = img_index

  def __getitem__(self, index):
    im = Image.open(self.img_names[index])
    if im.getbands()[0] == 'L':
      im = im.convert('RGB')
    im = self.transform(im)
    if im.shape != (3,224,224):
      print(self.img_names[index])

    class_index = self.img_index[index]
    attributes = self.predicate_binary_mat[class_index,:]
    if self.attributes:
      return im, class_index, attributes, self.img_names[index]
    else:
      return im, class_index
      
  def __len__(self):
    return len(self.img_names)

if __name__ == '__main__':

  trainset = AnimalDataset("/home/lorki/data", 'allclasses.txt', True, True)
  train_dataloader = DataLoader(trainset, batch_size=16, shuffle=True)
  testset = AnimalDataset("/home/lorki/data", 'allclasses.txt', True, False)
  test_dataloader = DataLoader(testset, batch_size=16, shuffle=False)
  print(len(test_dataloader))