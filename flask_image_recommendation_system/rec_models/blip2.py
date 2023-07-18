import glob
import io
import os
import pickle
import urllib.request

import numpy as np
import pandas as pd
import torch
from lavis.models import load_model_and_preprocess
from tqdm import tqdm

from const.pathname import *


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


# setup device to use
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
                                                                  model_type="pretrain", is_eval=True,
                                                                  device='cpu')
print('model loading completed')

# datasets path
datasets_path = 'unsplash-research-dataset-lite-latest/'
documents = ['photos']
datasets = {}

# load datasets
for doc in documents:
    files = glob.glob(datasets_path + doc + ".tsv*")

    subsets = []
    for filename in files:
        df = pd.read_csv(filename, sep='\t', header=0)['photo_image_url']
        subsets.append(df)

    datasets[doc] = pd.concat(subsets, axis=0, ignore_index=True)
print("Datasets load successfully")

# load embeddings
features_25k_image_list = []
with open('embeddings/features_25k_image_list.pkl', 'rb') as f:
    # features_25k_image_list = torch.load(f, map_location=torch.device('cpu'))
    features_25k_image_list = CPU_Unpickler(f).load()
print("Feature Embeddings load successfully")


def recommend_images_to_files_list(image):
    raw_image = image.convert('RGB').resize((596, 437))
    print('image resize completed')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    features_query_image = model.extract_features({'image': image}, mode='image')

    rank_image_index = rank_6(features_query_image.image_embeds_proj)

    count = 1
    image_file_list = []
    for url in datasets['photos'].loc[rank_image_index]:
        urllib.request.urlretrieve(url, os.path.join(OUTPUT_REC_DIR, f"output_image_{count}.jpeg"))
        image_file_list.append(os.path.join(OUTPUT_REC_DIR, f"output_image_{count}.jpeg"))
        count += 1
    return image_file_list


def recommend_images_to_urls(image):
    raw_image = image.convert('RGB').resize((596, 437))
    print('image resize completed')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    features_query_image = model.extract_features({'image': image}, mode='image')

    rank_image_index = rank_6(features_query_image.image_embeds_proj)

    return datasets['photos'].loc[rank_image_index].tolist()


def recommend_text_to_files_list(text):
    text = txt_processors["eval"](text)
    features_query_text = model.extract_features({'text_input': [text]}, mode='text')

    rank_image_index = rank_6(features_query_text.text_embeds_proj)

    count = 1
    image_file_list = []
    for url in datasets['photos'].loc[rank_image_index]:
        urllib.request.urlretrieve(url, os.path.join(OUTPUT_REC_DIR, f"output_image_{count}.jpeg"))
        image_file_list.append(os.path.join(OUTPUT_REC_DIR, f"output_image_{count}.jpeg"))
        count += 1
    return image_file_list


def recommend_text_to_urls(text):
    text = txt_processors["eval"](text)
    features_query_text = model.extract_features({'text_input': [text]}, mode='text')

    rank_image_index = rank_6(features_query_text.text_embeds_proj)

    return datasets['photos'].loc[rank_image_index].tolist()


def rank_6(feature_embeddings):
    print('query image features extraction completed')
    # image to image searching
    score_list = []

    for i in tqdm(range(len(features_25k_image_list))):
        feature = features_25k_image_list[i]
        if feature is None:
            score_list.append(0)
            continue
        similarity = (feature_embeddings @ feature[:, 0, :].t()).max()
        score_list.append(similarity)
    score = np.array(score_list)
    rank_image_index = np.argsort(-score)[0:6]
    return rank_image_index


def random_6(start, end, size):
    return np.random.randint(low=start, high=end, size=size)


def get_random_image_urls():
    return datasets['photos'].loc[random_6(0, len(datasets['photos']), 6)].tolist()
