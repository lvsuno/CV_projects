#%%
from pickle import load
from numpy import argmax
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input
#from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M
#from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
from utils import extract_features, generate_desc, cleanup_summary
import tensorflow as tf
#%%
# load the model
pretrained_model = EfficientNetV2M()
# Re-structure the model
pretrained_model= Model(inputs=pretrained_model.inputs, outputs=pretrained_model.layers[-2].output)
# extract the features from image
with tf.device('/cpu:0'):
    feature = extract_features('dog.jpeg',pretrained_model, preprocess_input)

# load tokenizer
with open('tokenizer_maxlength.pkl', 'rb') as f:
    tokenizer, max_length = load(f)

# Load captioning model
model = load_model('model_EfficientNetV2M_gru2.h5')

with tf.device('/cpu:0'):
    description = generate_desc(model, tokenizer, feature, max_length)

description = cleanup_summary(description)
print(description)
#%%