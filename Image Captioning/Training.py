#%% Import librairies
from pickle import load, dump

import tensorflow
from utils import create_sequences, load_set, load_clean_descriptions, load_photo_features, create_tokenizer, max_length
from tensorflow.keras.callbacks import ModelCheckpoint
from captioning_models import define_model
#%%
# Load training dataset 
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print(f'Training Dataset : {len(train)} instances')
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print(f'Descriptions : train= {len(train_descriptions)}')
# photo features
train_features = load_photo_features('features_EfficientNetV2M.pkl', train)
print(f'Photos: train= {len(train_features)}')
# %%
# prepare tokenizer
""" tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print(f'Vocabulary Size: {vocab_size}') """

#  determine the maximum sequence length
""" max_length = max_length(train_descriptions)
with open('tokenizer_maxlength.pkl', 'wb') as f:
    dump([tokenizer, max_length], f) 
print(f'Description Length: {max_length}') """

# Directly load the tokenizer if it's already saved
with open('tokenizer_maxlength.pkl', 'rb') as f:
    tokenizer, max_length = load(f)
vocab_size = len(tokenizer.word_index) + 1
#%% Prepare sequences
X1train, X2train, y_train = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)
#%% load validation set
filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
val = load_set(filename) 
print(f'Validation Dataset : {len(val)} instances')
# description
val_descriptions = load_clean_descriptions('descriptions.txt', val)
print(f'Descriptions : validation= {len(val_descriptions)}')
# photo features
val_features = load_photo_features('features_EfficientNetV2M.pkl', val)
print(f'Photos: validation= {len(val_features)}')
# prepare sequence
X1test, X2test, y_test = create_sequences(tokenizer, max_length, val_descriptions, val_features, vocab_size)

#%% Define model
model = define_model(vocab_size, max_length)
# define checkpoint callback
checkpoint = ModelCheckpoint('model_EfficientNetV2M_gru1.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
with tensorflow.device('/cpu:0'):
    model.fit([X1train, X2train], y_train, epochs=20, verbose=1, callbacks=[checkpoint], validation_data=([X1test, X2test], y_test))

#%%