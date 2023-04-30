import argparse
from pathlib import Path
import sys,os
sys.path.append('../layoutlm')
from LayoutLM.data_reader import prepare_data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import LayoutLMTokenizer, LayoutLMForSequenceClassification, AdamW
from LayoutLM.data_prep import normalize_box,apply_ocr,training_dataloader_from_df,encode_training_example
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D



dataset_path = "./sample"
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

train_data, valid_data = train_test_split(data, test_size=0.1, random_state=0, stratify=data.label)
train_data = train_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)
print(f"{len(train_data)} training examples, {len(valid_data)} validation examples")



# we need to define the features ourselves as the bbox of LayoutLM are an extra feature
training_features = Features({
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'label': ClassLabel(names=list(idx2label.keys())),
    'image_path': Value(dtype='string'),
    'words': Sequence(feature=Value(dtype='string')),
})

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")


train_dataloader = training_dataloader_from_df(train_data,training_features)
valid_dataloader = training_dataloader_from_df(valid_data,training_features)

