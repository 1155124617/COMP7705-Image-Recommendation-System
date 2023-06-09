import torch
import urllib.request
import numpy as np
import pandas as pd
import pdb

from PIL import Image
from lavis.models import load_model_and_preprocess

# setup device to use
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
model, vis_processors, _  = load_model_and_preprocess(name="blip2_feature_extractor",
                                                                  model_type="pretrain", is_eval=True,
                                                                  device='cpu')
print('model loading completed')

print('start reading csv file')
df_sample = pd.read_csv('image_recommend/data/20sampleimages.csv')
print('read samples completed')

image_list = list(df_sample['photo_image_url'])
sample_list = []
print(f'start downloading samples, sample size : {len(image_list)}')
for image_url in image_list:
    urllib.request.urlretrieve(image_url, "gfg.png")
    raw_image = Image.open("gfg.png").convert("RGB").resize((596, 437))
    print('image downloaded')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    print('image preprocessed')
    sample_list.append({"image": image})

print('start extracting features')
features_image_list = []
for si in sample_list:
    features_image_list.append(model.extract_features(si, mode='image').image_embeds_proj)
    print('feature extracted')
# features_image_list = [model.extract_features(si, mode="image").image_embeds_proj for si in sample_list]


def recommend_images(image):
    raw_image = image.convert('RGB').resize((596, 437))

    print('image resize completed')

    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    features_query_image = model.extract_features({'image': image}, mode='image')
    print('query image features extraction completed')

    # image to image searching
    score_list = []
    for feature in features_image_list:
        similarity = (features_query_image.image_embeds_proj @ feature[:, 0, :].t()).max()
        score_list.append(similarity)

    score = np.array(score_list)
    rank_image = np.argsort(-score)[0]

    return df_sample[rank_image]
