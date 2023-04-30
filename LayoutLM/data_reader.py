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

    return data, idx2label, label2idx

