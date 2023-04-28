import os
import pandas as pd
from sklearn.model_selection import train_test_split

import argparse
from pathlib import Path

def prepare_data(dataset_path):
    labels = [label for label in os.listdir(dataset_path)]
    idx2label = {v: k for v, k in enumerate(labels)}
    label2idx = {k: v for v, k in enumerate(labels)}

    images = []
    labels = []

    for label in os.listdir(dataset_path):
        images.extend([
            f"{dataset_path}/{label}/{img_name}" for img_name in os.listdir(f"{dataset_path}/{label}")
        ])
        labels.extend([
            label for _ in range(len(os.listdir(f"{dataset_path}/{label}")))
        ])
    data = pd.DataFrame({'image_path': images, 'label': labels})

    train_data, valid_data = train_test_split(data, test_size=0.09, random_state=0, stratify=data.label)
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    print(f"{len(train_data)} training examples, {len(valid_data)} validation examples")
    
    return train_data, valid_data, idx2label

