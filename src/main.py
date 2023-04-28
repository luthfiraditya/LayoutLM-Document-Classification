import argparse
from pathlib import Path
import sys,os
sys.path.append('../layoutlm')
from LayoutLM.data_reader import prepare_data
from LayoutLM.data_prep import normalize_box,apply_ocr,encode_training_example,training_dataloader_from_df
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D

dataset_path = "./small_data"
train_data, valid_data, idx2label = prepare_data(dataset_path)

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


train_dataloader = training_dataloader_from_df(train_data)