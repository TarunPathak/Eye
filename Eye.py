__author__='Tarun Pathak'

#importing libraries
import os
import glob
import numpy
from nltk.corpus import wordnet
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.applications.imagenet_utils import decode_predictions,preprocess_input

#function to return synonyms
#synonym same as current word are ignored
def get_synonyms(word):
    synonyms=''
    for syn in wordnet.synsets(word):
        for lma in syn.lemmas():
            if not str(lma.name()).lower()==str(word).lower():
                if len(synonyms)==0:
                    synonyms=synonyms + lma.name()
                else:
                    synonyms = synonyms + ', ' + lma.name()

    return synonyms

#function to generate image tags based on
#predictions from neural network
def generate_tags(prediction):
    #variable
    tags=''
    #filtering predicitions with 60% or more probability
    #tags along with their synonyms are returned
    for x in range(0,5):
        if float(prediction[0][x][2])*100>=50:
            #getting synonyms
            if len(tags)==0:
                tags=str(prediction[0][x][1]) + ',' + get_synonyms(str(prediction[0][x][1]))
            else:
                tags = tags + ', '+ str(prediction[0][x][1]) + ',' + get_synonyms(str(prediction[0][x][1]))

    #returning tags
    return tags

#function to classify image
def get_predictions(model,imagepath):
    #loading image
    #converting to array and preprocessing image
    img=image.load_img(path=os.path.abspath(imagepath),target_size=(224,224))
    img_arr=image.img_to_array(img)
    img_arr=numpy.expand_dims(img_arr,axis=0)
    img_arr=preprocess_input(img_arr)
    #prediciton
    pred=model.predict(img_arr)
    return decode_predictions(pred)

#processing images
def process_images():
    # loading model (neural network)
    model = ResNet50(weights='imagenet')
    # iterates through images
    for fileobject in os.listdir('.'):
        if str(os.path.splitext(fileobject)[1]).lower() == '.jpg':
            print('processing ' +  fileobject)
            print('identifying objects in the image.')
            #getting predictions from neural network
            prediction=get_predictions(model,fileobject)
            #generating tags from prediction
            generate_tags(prediction)
            print('---------------------------------------')

#main
if __name__=='__main__':
    #count of images (jpg) in current folder
    filecount=len(glob.glob('./*.jpg'))
    #exiting (if no imges found)
    #else classifying images
    if filecount==0:
        print('No JPG images found. The program will now terminate.')
    else:
        process_images()