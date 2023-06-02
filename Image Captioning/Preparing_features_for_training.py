#%% 
from os import listdir
from pickle import dump
#from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils  import img_to_array
#from tensorflow.keras.applications.vgg16 import preprocess_input
#from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tqdm import tqdm
from utils import load_descriptions, load_doc, clean_descriptions, to_vocabulary, save_descriptions
#tensorflow.keras.utils.image_dataset_from_directory
# %%
# Extract features from each photo in the directory
def extract_features(directory, model):
    # load the model
    # model = VGG16()
    # extract features from each photo
    features = dict()
    for name in tqdm(listdir(directory)):
        # load image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(480, 480))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
        print(f"{name}")
    return features

directory = 'Flicker8k_Dataset'
#model = VGG19()
model = EfficientNetV2M()
#%%
# re-structure the model
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# summarize
model.summary()
features = extract_features(directory, model)
print(f'Extracted Features: {len(features)}')
#%%
# save to file
dump(features, open('features_EfficientNetV2M.pkl', 'wb'))

#%% Prepare Text Data
filename = 'Flickr8k_text/Flickr8k.token.txt'
# load descriptions
doc = load_doc(filename)
# parse descriptions
descriptions = load_descriptions(doc)
print(f'Loaded: {len(descriptions)}')
# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print(f'Vocabulary size: {len(vocabulary)}')
# Save clean description to file
save_descriptions(descriptions, 'descriptions.txt')
# %%
