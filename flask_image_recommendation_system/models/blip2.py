import pickle

import torch
import urllib.request
import numpy as np
import pandas as pd
import glob
import io

from tqdm import tqdm
from PIL import Image
from lavis.models import load_model_and_preprocess

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

'''print('start reading csv file')
df_sample = pd.read_csv('data/20sampleimages.csv')
print('read samples completed')

image_list = list(df_sample['photo_image_url'])
sample_list = []
print(f'start downloading samples, sample size : {len(image_list)}')
for image_url in image_list:
    urllib.request.urlretrieve(image_url, "temp/gfg.png")
    raw_image = Image.open("temp/gfg.png").convert("RGB").resize((596, 437))
    print('image downloaded')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    print('image preprocessed')
    sample_list.append({"image": image})

print('start extracting features')
features_image_list = []
for si in sample_list:
    features_image_list.append(model.extract_features(si, mode='image').image_embeds_proj)
    print('feature extracted')'''
# features_image_list = [model.extract_features(si, mode="image").image_embeds_proj for si in sample_list]


def recommend_images_to_files_list(image):
    rank_image_index = rank_6(image)

    count = 1
    image_file_list = []
    for url in datasets['photos'].loc[rank_image_index]:
        urllib.request.urlretrieve(url, f"temp/output_image_{count}.jpeg")
        image_file_list.append(f"temp/output_image_{count}.jpeg")
        count += 1
    return image_file_list


def recommend_images_to_urls(image):
    rank_image_index = rank_6(image)

    return datasets['photos'].loc[rank_image_index].tolist()


def rank_6(image):
    raw_image = image.convert('RGB').resize((596, 437))
    print('image resize completed')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    features_query_image = model.extract_features({'image': image}, mode='image')
    print('query image features extraction completed')
    # image to image searching
    score_list = []

    for i in tqdm(range(len(features_25k_image_list))):
        feature = features_25k_image_list[i]
        if feature is None:
            score_list.append(0)
            continue
        similarity = (features_query_image.image_embeds_proj @ feature[:, 0, :].t()).max()
        score_list.append(similarity)
    score = np.array(score_list)
    rank_image_index = np.argsort(-score)[0:6]
    return rank_image_index
