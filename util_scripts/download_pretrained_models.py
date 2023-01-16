import os

import wget

BASE_URL = r"https://cvg.hhi.fraunhofer.de/CASAPose/"
DOWNLOAD_PATH = r"data/pretrained_models"

LMO_MODEL_PATH = BASE_URL + r"result_w_8.h5"
LM_MODEL_PATH = BASE_URL + r"result_w_13.h5"

if not os.path.exists(DOWNLOAD_PATH):
    os.makedirs(DOWNLOAD_PATH)

LMO_MODEL_PATH_OUT = os.path.join(DOWNLOAD_PATH, "result_w_8.h5")
LM_MODEL_PATH_OUT = os.path.join(DOWNLOAD_PATH, "result_w_13.h5")

if not os.path.exists(LMO_MODEL_PATH_OUT):
    wget.download(LMO_MODEL_PATH, out=LMO_MODEL_PATH_OUT)

if not os.path.exists(LM_MODEL_PATH_OUT):
    wget.download(LMO_MODEL_PATH, out=LM_MODEL_PATH_OUT)
