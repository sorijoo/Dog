import streamlit as st 
import os
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from imgaug import augmenters as iaa

from sklearn.model_selection import train_test_split

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *
from keras.callbacks import *

from keras.applications.densenet import DenseNet121, preprocess_input

import cv2

st.title("GUESS WHAT DOG YOU ARE")

st.write("""
# Put your picture and see what is your nearest breed!
""")
filename = st.file_uploader("Choose a file")

breed_list = ['n02093256-Staffordshire_bullterrier',
 'n02094114-Norfolk_terrier',
 'n02086079-Pekinese',
 'n02088364-beagle',
 'n02107312-miniature_pinscher',
 'n02113624-toy_poodle',
 'n02097130-giant_schnauzer',
 'n02095889-Sealyham_terrier',
 'n02101556-clumber',
 'n02113712-miniature_poodle',
 'n02108000-EntleBucher',
 'n02098105-soft-coated_wheaten_terrier',
 'n02110958-pug',
 'n02088094-Afghan_hound',
 'n02099849-Chesapeake_Bay_retriever',
 'n02111889-Samoyed',
 'n02109047-Great_Dane',
 'n02097474-Tibetan_terrier',
 'n02102973-Irish_water_spaniel',
 'n02094258-Norwich_terrier',
 'n02102480-Sussex_spaniel',
 'n02086646-Blenheim_spaniel',
 'n02097658-silky_terrier',
 'n02090379-redbone',
 'n02092339-Weimaraner',
 'n02091831-Saluki',
 'n02096294-Australian_terrier',
 'n02086240-Shih-Tzu',
 'n02108551-Tibetan_mastiff',
 'n02086910-papillon',
 'n02096177-cairn',
 'n02094433-Yorkshire_terrier',
 'n02110627-affenpinscher',
 'n02108422-bull_mastiff',
 'n02107908-Appenzeller',
 'n02096051-Airedale',
 'n02088466-bloodhound',
 'n02100735-English_setter',
 'n02106382-Bouvier_des_Flandres',
 'n02112018-Pomeranian',
 'n02096437-Dandie_Dinmont',
 'n02107574-Greater_Swiss_Mountain_dog',
 'n02105251-briard',
 'n02097047-miniature_schnauzer',
 'n02090721-Irish_wolfhound',
 'n02100877-Irish_setter',
 'n02100236-German_short-haired_pointer',
 'n02113799-standard_poodle',
 'n02113978-Mexican_hairless',
 'n02090622-borzoi',
 'n02101388-Brittany_spaniel',
 'n02106550-Rottweiler',
 'n02085936-Maltese_dog',
 'n02110806-basenji',
 'n02097209-standard_schnauzer',
 'n02105162-malinois',
 'n02106166-Border_collie',
 'n02104365-schipperke',
 'n02089078-black-and-tan_coonhound',
 'n02093991-Irish_terrier',
 'n02088632-bluetick',
 'n02098413-Lhasa',
 'n02091032-Italian_greyhound',
 'n02109961-Eskimo_dog',
 'n02111500-Great_Pyrenees',
 'n02097298-Scotch_terrier',
 'n02100583-vizsla',
 'n02107683-Bernese_mountain_dog',
 'n02089867-Walker_hound',
 'n02116738-African_hunting_dog',
 'n02099601-golden_retriever',
 'n02113186-Cardigan',
 'n02101006-Gordon_setter',
 'n02093754-Border_terrier',
 'n02115641-dingo',
 'n02098286-West_Highland_white_terrier',
 'n02110063-malamute',
 'n02099267-flat-coated_retriever',
 'n02099429-curly-coated_retriever',
 'n02091134-whippet',
 'n02105855-Shetland_sheepdog',
 'n02091467-Norwegian_elkhound',
 'n02093428-American_Staffordshire_terrier',
 'n02105505-komondor',
 'n02106662-German_shepherd',
 'n02105056-groenendael',
 'n02085782-Japanese_spaniel',
 'n02085620-Chihuahua',
 'n02112137-chow',
 'n02095570-Lakeland_terrier',
 'n02106030-collie',
 'n02093647-Bedlington_terrier',
 'n02108915-French_bulldog',
 'n02087046-toy_terrier',
 'n02102040-English_springer',
 'n02110185-Siberian_husky',
 'n02091244-Ibizan_hound',
 'n02102318-cocker_spaniel',
 'n02108089-boxer',
 'n02111129-Leonberg',
 'n02102177-Welsh_springer_spaniel',
 'n02111277-Newfoundland',
 'n02113023-Pembroke',
 'n02087394-Rhodesian_ridgeback',
 'n02091635-otterhound',
 'n02096585-Boston_bull',
 'n02093859-Kerry_blue_terrier',
 'n02099712-Labrador_retriever',
 'n02104029-kuvasz',
 'n02105412-kelpie',
 'n02115913-dhole',
 'n02092002-Scottish_deerhound',
 'n02095314-wire-haired_fox_terrier',
 'n02105641-Old_English_sheepdog',
 'n02088238-basset',
 'n02109525-Saint_Bernard',
 'n02112350-keeshond',
 'n02089973-English_foxhound',
 'n02107142-Doberman',
 'n02112706-Brabancon_griffon']


num_classes = len(breed_list)


label_maps = {}
label_maps_rev = {}

for i, v in enumerate(breed_list):
    label_maps.update({v: i})
    label_maps_rev.update({i : v})

def create_functional_model():

    inp = Input((224, 224, 3))
    backbone = DenseNet121(input_tensor=inp,
                           weights="imagenet",
                           include_top=False)
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    outp = Dense(num_classes, activation="softmax")(x)
#     model = Model(inp, outp)
    model = Model(inp, outp)
    
    model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["acc"])
    
    for layer in model.layers[:]:
        layer.trainable = True
    
    return model    

pretrained_model = create_functional_model()

pretrained_model.load_weights("dog_breed_classifier_model.h5")

def upload_and_predict2(filename):
    img = Image.open(filename)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    print(img.size)
    # show image
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    # predict
#     img = imread(filename)
#     img = preprocess_input(img)
    probs = pretrained_model.predict(np.expand_dims(img, axis=0))
    for idx in probs.argsort()[0][::-1][:8]:
        print("{:.2f}%".format(probs[0][idx]*100), "\t", label_maps_rev[idx].split("-")[-1])

if filename is not None:
    img = Image.open(filename)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')

    probs = pretrained_model.predict(np.expand_dims(img, axis=0))
    # text = []
    st.image(img, use_column_width=True)
    for idx in probs.argsort()[0][::-1][:8]:
        st.text("{:.2f}%".format(probs[0][idx]*100) +" "+ label_maps_rev[idx].split("-")[-1])

