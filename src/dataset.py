import os
import sys
from collections import defaultdict

# path of current file
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)

# torch
import torch

from torch.utils.data import Dataset

from src.utils import apply_ocr, encode_document

class DocumentDataset(Dataset):
    """Document dataset."""

    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (string): Path to the data folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # define the paths
        self.data_path = data_path
        self.images_folder = os.path.join(self.data_path)
        self.ocr_folder = os.path.join(self.data_path, 'ocr')

        # labels
        self.labels = os.listdir(self.images_folder)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for idx, label in enumerate(self.labels)}

        # curate all the image paths
        self.file_names = defaultdict(list)
        for label in self.labels:
            label_folder = os.path.join(self.images_folder, label)
            image_paths = os.listdir(label_folder)
            for image_path in image_paths:
                file_name = image_path.split('/')[-1].split('.')[0]
                if 'ipynb_checkpoints' in image_path:
                    continue
                image_full_path = os.path.join(label_folder, image_path)
                self.file_names[file_name].append(image_full_path)

        # create the final dataset as a list of tuples
        self.dataset = []
        for key, val in self.file_names.items():
            self.dataset.append((key, val[0]))

        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name, image_path = self.dataset[idx]
        
        label = image_path.split('\\')[-2]


        # apply ocr
        ocr_output = apply_ocr(image_path)

        # encode the document
        encode_input = {
            'words': ocr_output['words'],
            'nboxes': ocr_output['nboxes'],
            'aboxes': ocr_output['aboxes'], 
            'label': self.label_to_idx[label]
        }
        encode_output = encode_document(encode_input)

        sample = {
            'file_name': file_name,
            'image_path': image_path,
            'label': self.label_to_idx[label],
            'words': ocr_output['words'],
            'bbox': encode_output['bbox'],
            'aboxes': ocr_output['aboxes'],
            'input_ids': encode_output['input_ids'],
            'attention_mask': encode_output['attention_mask'],
            'token_type_ids': encode_output['token_type_ids'],
        }

        return sample