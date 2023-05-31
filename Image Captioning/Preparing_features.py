#%% 
from os import listdir
from os import path
from pickle import dump
from pyexpat import features
from keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils  import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from tqdm import tqdm
#tensorflow.keras.utils.image_dataset_from_directory
# %%
# Extract features from each photo in the directory
def extract_features(directory):
    # load the model
    model = VGG16()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # summarize
    model.summary()
    # extract features from each photo
    features = dict()
    for name in tqdm(listdir(directory)):
        # load image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for VGG model
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
features = extract_features(directory)
print(f'Extracted Features: {len(features)}')
#%%
# save to file
dump(features, open('features.pkl', 'wb'))

#%% Prepare Text Data
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# extract descriptions for images
def load_descriptions(doc):
    mapping = dict()
    # process lines
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # remove filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # Create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
            # store description
            mapping[image_id].append(image_desc)
    return mapping

# Clean descriptions
def clean_descriptions(descriptions):
    # prepare regex for char filtering
    

# %%
filename = 'Flickr8k_text/Flickr8k.token.txt'
# load descriptions
doc = load_doc(filename)
# parse descriptions
descriptions = load_descriptions(doc)
print(f'Loaded: {len(descriptions)}')