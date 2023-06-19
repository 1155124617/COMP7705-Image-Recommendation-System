import numpy as np
import pandas as pd

def recommend_images_to_files_list(image):
    return ['path1', 'path2']


def recommend_images_to_urls(image):
    return pd.DataFrame({"URL": ['url1', 'url2'], "others": [1, 2]})["URL"].tolist()