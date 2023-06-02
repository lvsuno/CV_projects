from utils import evaluate_model, load_set, load_clean_descriptions, create_tokenizer,  load_photo_features
from tensorflow.keras.models import load_model
import tensorflow as tf
from pickle import load

# %%
# load tokenizer and max_length
with open('tokenizer_maxlength.pkl', 'rb') as f:
    tokenizer, max_length = load(f)

#%% load testing dataset
filename = 'Flickr8k_text/Flickr_8k.testImages.txt'
test = load_set(filename) 
print(f'Testing Dataset : {len(test)} instances')
# description
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print(f'Descriptions : test= {len(test_descriptions)}')
# photo features
test_features = load_photo_features('features_EfficientNetV2M.pkl', test)
print(f'Photos: train= {len(test_features)}')

#%% Load the model
filename = 'model_EfficientNetV2M_gru2.h5'
model = load_model(filename)
# evaluate model
with tf.device('/cpu:0'):
    evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
# %%