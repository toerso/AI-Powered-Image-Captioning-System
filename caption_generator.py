
import os
import keras
from keras.preprocessing.text import tokenizer_from_json
import numpy as np
import matplotlib.pyplot as plt
from build_model.build import feature_extractions, sample_caption
import json
from pickle import load, dump
    
with open('tokenizer.json', 'r') as f:
    tokenizer_json = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_json)
    
model = keras.models.load_model("./sample_model.h5")
vocab_size = tokenizer.num_words
max_length = 37

#sampling
features = feature_extractions("./sample_images")

for idx, filename in enumerate(features.keys()):
    plt.figure(idx+1)
    caption = sample_caption(model, tokenizer, max_length, vocab_size, features[filename])
    
    image = keras.preprocessing.image.load_img("./sample_images/{fn}.jpg".format(fn=filename))
    plt.imshow(image)
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)



