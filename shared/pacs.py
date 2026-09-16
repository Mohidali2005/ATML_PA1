"""
This file loads the pacs dataset from its huggingface parquet mirror and
turns each row into a decoded image so task two and task three can use
it without touching huggingface again after the first download
"""

import io
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

DOMAINS = ["photo","art_painting","cartoon","sketch"]
CLASSES = ["dog","elephant","giraffe","guitar","horse","house","person"]

def load_pacs_dataframe():
    """
    This function downloads the pacs parquet file the first time it is
    called and returns a pandas dataframe with one row per image and
    columns for the raw image bytes the domain name and the class label
    """
    path = hf_hub_download(repo_id="flwrlabs/pacs",repo_type="dataset",filename="data/train-00000-of-00001.parquet")
    return pd.read_parquet(path)

def decode_image(image_field):
    """
    This function turns one row of the image column into a regular rgb
    pillow image
    """
    return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")
