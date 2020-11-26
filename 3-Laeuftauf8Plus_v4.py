#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:11:31 2020

@author: base

"""
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import os
from pathlib import Path
from time import time
import concurrent.futures
import sys

print('Tensorflowversion: ' + tf.__version__)
print('You work here: ', Path.cwd())

# Check System Architecture
#-----------------------------------------------------------------------------
import subprocess
Architekture=((subprocess.check_output("lscpu | grep Architecture ", shell=True).strip()).decode())
if "aarch46" in Architekture:
    Runningsystem='ARM'
elif 'x86_64' or 'x86_32' in Architekture:
    Runningsystem='PC'

# Define Variables
#-----------------------------------------------------------------------------
howtoface=['mtcnn','openCV']
qunatornot=['quant','tfl']
backendchoise=['cpu','npu']
howtoloadmodel=['armNN','TL']
valuetype=['float','int']

facedetect=howtoface[1]     #0=mtcnn 1=openCV facecascade
modeltype=qunatornot[0]     #0= quant  1= tflite
Loadtype=howtoloadmodel[1]  #0=armNN 1=tflite
prefbackend=backendchoise[0]#0=cpu 1=NPU
inputtype=valuetype[0]      #0=float 1= int
largeImg=True

ImgSize=(500,500) #Size of the plottet image IF largeImg IS SET TRUE
VideoDevice=1 # '1:Webcam or 2:VM016
video_input_device=0
Gesichter= False  # either True for only croped celebrity faces or False for original celbrity image. DEciding which images to show. Cropped or total resized image
brt = 90  # value could be + or - for brightness or darkness
gray=False
p=35# frame size around detected face
width=height=224 # size of the cropped image. Same as required for network
mitte=np.empty(shape=[0, 0])
mittleres_Gesicht_X=()

if Runningsystem =='PC':
    cascaderpath='Cascader'
    modelpath='models/'
    embeddingpath='Data/Embeddings/'
    
if Runningsystem =='ARM':
    cascaderpath='Cascader/'
    modelpath='models/tflite'
    embeddingpath='Embeddings/'


#TFLITE int8 QUANTIZED MODELS 
#*******************

inputtype=valuetype[0]      #0=float 1= int
embeddingsfile='EMBEDDINGS_quantized_modelh5-15.json'
model="quantized_modelh5-15.tflite"

model_path=str(Path.cwd() / modelpath / model)

#Load face cascader 
#-----------------------------------------------------------------------------

#face_cascade = cv2.CascadeClassifier(str(Path.cwd() / cascaderpath / 'haarcascade_frontalface_alt.xml'))
#face_cascade = cv2.CascadeClassifier(str(Path.cwd() / cascaderpath / 'lbpcascade_frontalface.xml'))
face_cascade = cv2.CascadeClassifier(str(Path.cwd() / cascaderpath / 'lbpcascade_frontalface_improved.xml'))
print('cascader loaded  ...')
    
# Load Model tflite
#-----------------------------------------------------------------------------
# Load TFLite model and allocate tensors.Beide modelle funktionieren
warm=time()
if Loadtype=='TL':
    try:  
        interpreter = tflite.Interpreter(model_path)
        #interpreter = tf.lite.Interpreter(model_path)# This works if the flite interpreter cannot be installed. This is the only TF version
    except ValueError as e:
        print("Error: Modelfile could not be found. Check if you are in the correct workdirectory. Errormessage:  " + str(e))
        #Depending on the version of TF running, check where lite is set :
        if tf.__version__.startswith ('1.'):
            print('lite in dir(tf.contrib)' + str('lite' in dir(tf.contrib)))
    
        elif tf.__version__.startswith ('2.'):
            print('lite in dir(tf)? ' + str('lite' in dir(tf)))
        print('workdir: ' + os.getcwd())
        sys.exit()
    
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details() 
    
    print('model loaded with tflite...', time()-warm)

# Load Model armNN DOES NOT WORK YET
#-----------------------------------------------------------------------------
elif Loadtype=='armNN':
    import pyarmnn as ann
    
    # ONNX, Caffe and TF parsers also exist.
    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile(model_path)
    #Get the input binding information by using the name of the input layer.
    input_binding_info = parser.GetNetworkInputBindingInfo(0, 'model/input')

    # Create a runtime object that will perform inference.
    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)
    #Choose preferred backends for execution and optimize the network.
    # Backend choices earlier in the list have higher preference.
    if prefbackend=='cpu':
        preferredBackends = [ann.BackendId('CpuRef')]
    elif prefbackend=='npu':
        preferredBackends = [ann.BackendId('CpuAcc')]
    opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())
    
    # Load the optimized network into the runtime.
    net_id, _ = runtime.LoadNetwork(opt_network)


# Load Embeddings
#-----------------------------------------------------------------------------
##LOADING CSV (easier with pandas, but 8M Plus does not yet support pandas)
emb=time()
import json 

f = open((Path.cwd() / embeddingpath  / embeddingsfile),'r') 
ImportedData =json.load(f)
dataE=[np.array(ImportedData['Embedding'][str(i)]) for i in range(len(ImportedData['Name']))]
dataN=[np.array(ImportedData['Name'][str(i)]) for i in range(len(ImportedData['Name']))]
dataF=[np.array(ImportedData['File'][str(i)]) for i in range(len(ImportedData['Name']))]

print('Embeddings loaded      ...',time()-emb)

#Define functions
#-----------------------------------------------------------------------------
def preprocess_input(x, data_format, version): #Choose version same as in " 2-Create embeddings database.py or jupyter"
    x_temp = np.copy(x)
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
            
    elif version == 3:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= np.round(91.4953).astype('uint8')
            x_temp[:, 1, :, :] -= np.round(103.8827).astype('uint8')
            x_temp[:, 2, :, :] -= np.round(131.0912).astype('uint8')
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= np.round(91.4953).astype('uint8')
            x_temp[..., 1] -= np.round(103.8827).astype('uint8')
            x_temp[..., 2] -= np.round(131.0912).astype('uint8')
    else:
        raise NotImplementedError

    return x_temp

def splitDataFrameIntoSmaller(df, chunkSize):
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf

def faceembeddingNP(YourFace,CelebDaten):
    Dist=[]
    for i in range(len(CelebDaten)):
        Celebs=np.array(CelebDaten[i]) 
        Dist.append(np.linalg.norm(YourFace-Celebs))
    return Dist

print('functions defined         ...')
# Split data for threadding
#-----------------------------------------------------------------------------
splitt=time()
celeb_embeddings=splitDataFrameIntoSmaller(dataE, int(np.ceil(len(dataE)/4)))   
# celeb_Names=splitDataFrameIntoSmaller(dataN, int(np.ceil(len(dataN)/4)))   
# celeb_File=splitDataFrameIntoSmaller(dataF, int(np.ceil(len(dataF)/4)))   
print('Embeddings split             ...' ,time()-splitt)

 
#open Camera, Get frame middel for frame optimization
#-----------------------------------------------------------------------------
if VideoDevice == 1:
    cap= cv2.VideoCapture(video_input_device)

if not cap.isOpened():
    print('Error: VideoCapture not opened')
    sys.exit(0)

elif VideoDevice == 2:
    buildinfo = cv2.getBuildInformation()

    if buildinfo.find("GStreamer") < 0:
        print('no GStreamer support in OpenCV')
        exit(0)

    #can be changed to e.g. 640x480
    #width=1024
    #height=760

    width=1280
    height=800

    #Set g stremaer pipeline
    #Weißabgleich
    cmd="v4l2-ctl -d /dev/video4 --set-ctrl=autogain_digital=0"
    os.system(cmd)
    cmd="v4l2-ctl -d /dev/video4 --set-ctrl=digital_gain_red=1000"
    os.system(cmd)
    cmd="v4l2-ctl -d /dev/video4 --set-ctrl=digital_gain_green_red=1300"
    os.system(cmd)
    cdm="v4l2-ctl -d /dev/video4 --set-ctrl=digital_gain_blue=1000"
    os.system(cmd)
    cmd="v4l2-ctl -d /dev/video4 --set-ctrl=digital_gain_green_blue=1500"
    os.system(cmd)
    cmd="v4l2-ctl -d /dev/video4 --set-ctrl=autogain_analogue=0"
    os.system(cmd)


    cmd = "v4l2-ctl -d /dev/video0 --set-fmt-video=pixelformat=GRBG,width={width},height={height}".format(width=width, height=height)
    os.system(cmd)
    cmd = "v4l2-ctl -d /dev/video0 --set-selection=target=crop,left=0,top=4,width={width},height={height}".format(width=width, height=height)
    os.system(cmd)

    pipeline = "v4l2src device=/dev/video0 ! video/x-bayer,format=grbg,depth=8,width={width},height={height} ! bayer2rgb ! videoconvert !videoscale ! video/x-raw,width=640,height=400 !  appsink".format(width=width, height=height)
    if Runningsystem =='PC':
        cap = cv2.VideoCapture(0)
    if Runningsystem =='ARM':
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


ret, frame = cap.read() 
framemitte=np.shape(frame)[1]/2
print('camera loaded                   ...')
print('pre-processing done                !!!')

#Start
#-----------------------------------------------------------------------------


while(True):
# CAPTURE FRAME BY FRAME    
    ret, frame = cap.read() 
    if gray==True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame=cv2.flip(frame,1)  
    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('frame', frame)

#DECTECT FACE IN VIDEO CONTINUOUSLY       
    faces_detected = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)#, Size(50,50))
    for (x,y,w,h) in faces_detected:
        rechteck=cv2.rectangle(frame, (x-p, y-p+2), (x+w+p, y+h+p+2), (0, 255, 0), 2)  
        #rechteck=cv2.rectangle(frame, (x-p, y-p+2), (x+int(np.ceil(height))+p, y+int(np.ceil(height))+p+2), (0, 0, 100), 2)  
        cv2.imshow('frame', rechteck)     

# DETECT KEY INPUT  - ESC OR FIND MOST CENTERED FACE  
    key = cv2.waitKey(1)
    if key == 27: #Esc key
        cap.release()
        cv2.destroyAllWindows()
        break
    if key ==32: 
        mittleres_Gesicht_X=()
        mitte=()
        if len(faces_detected) !=0: # only if the cascader detected a face, otherwise error
            start1 = time()
#FIND MOST MIDDLE FACE            
            for (x,y,w,h) in faces_detected:
                mitte=np.append(mitte,(x+w/2))               
            mittleres_Gesicht_X = (np.abs(mitte - framemitte)).argmin()
            print('detect middel face ' ,time()-start1)
# FRAME THE DETECTED FACE
            start2=time()
            #print(faces_detected[mittleres_Gesicht_X])
            (x, y, w, h) = faces_detected[mittleres_Gesicht_X]
            img=frame[y-p+2:y+h+p-2, x-p+2:x+w+p-2] #use only the detected face; crop it +2 to remove frame # CHECK IF IMAGE EMPTY (OUT OF IMAGE = EMPTY)     

            if len(img) != 0: # Check if face is out of the frame, then img=[], throwing error
                print('detect face ',time()-start2)

# CROP IMAGE 
                start3=time()
                if img.shape > (width,height): #downsampling
                    img_small=cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA) #resize the image to desired dimensions e.g., 256x256  
                elif img.shape < (width,height): #upsampling
                    img_small=cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC) #resize the image to desired dimensions e.g., 256x256                      
                cv2.imshow('frame',img_small)
                cv2.waitKey(1) #hit any key
                end3=time()
                print('face crop', end3-start3)
# IMAGE PREPROCESSING
                start4=time()
                if inputtype=='int':
                    samples = np.expand_dims(img_small, axis=0)
                    samples = preprocess_input(samples, data_format=None, version=3).astype('int8')#data_format= None, 'channels_last', 'channels_first' . If None, it is determined automatically from the backend
                else:
                    pixels = img_small.astype('float32')
                    samples = np.expand_dims( pixels, axis=0)
                    samples = preprocess_input(samples, data_format=None, version=2)#data_format= None, 'channels_last', 'channels_first' . If None, it is determined automatically from the backend
                #now using the tflight model
                print('preprocess data for model' , time()-start4)
# CREATE FACE EMBEDDINGS                
                if Loadtype=='armNN':
                    prep=time()
                    input_tensors = ann.make_input_tensors([input_binding_info], [samples])
                    # Get output binding information for an output layer by using the layer name.
                    output_binding_info = parser.GetNetworkOutputBindingInfo(0, 'model/output')
                    output_tensors = ann.make_output_tensors([output_binding_info])
                    runtime.EnqueueWorkload(0, input_tensors, output_tensors)
                    print('ANN preperation ',time()-prep)
                    start42=time()
                    EMBEDDINGS=ann.workload_tensors_to_ndarray(output_tensors)
                elif Loadtype=='TL':
                    prep=time()
                    input_shape = input_details[0]['shape']
                    input_data = samples
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    print('ANN preperation ',time()-prep)
                    start42=time()
                    EMBEDDINGS = interpreter.get_tensor(output_details[0]['index'])
                print('create face embeddings' , time()-start42)
# READ CELEB EMBEDDINGS AND COMPARE  
                start_EU=time()
                EuDist=[]
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    ergebniss_1=executor.submit(faceembeddingNP,EMBEDDINGS,np.array(celeb_embeddings[0]))
                    ergebniss_2=executor.submit(faceembeddingNP,EMBEDDINGS,np.array(celeb_embeddings[1]))
                    ergebniss_3=executor.submit(faceembeddingNP,EMBEDDINGS,np.array(celeb_embeddings[2]))
                    ergebniss_4=executor.submit(faceembeddingNP,EMBEDDINGS,np.array(celeb_embeddings[3]))

                if ergebniss_1.done() & ergebniss_2.done() & ergebniss_3.done() & ergebniss_4.done():
                    EuDist.extend(ergebniss_1.result())
                    EuDist.extend(ergebniss_2.result())
                    EuDist.extend(ergebniss_3.result())
                    EuDist.extend(ergebniss_4.result())
                print('Create_EuDist', time()-start_EU)

                start_Min=time()
                idx = np.argpartition(EuDist, 5)                
                folder_idx= dataN[idx[0]]
                image_idx = dataF[idx[0]] 
                #folder_idx= dataN[np.argmin(EuDist)]
                #image_idx = dataF[np.argmin(EuDist)] 
                print('find minimum for facematch', time()-start_Min)
                
# PLOT IMAGES       
                start6=time()
                path=Path.cwd()

                if Gesichter == False:
                    pfad=str(Path.cwd() / 'Data/sizeceleb_224_224' / str(folder_idx) / str(image_idx))
                elif Gesichter == True:
                    pfad=str(Path.cwd() / 'Data/Celebs_faces' / str(folder_idx) / str(image_idx))    
                    
                Beleb=cv2.imread(pfad)
                if np.shape(Beleb) != (width,height): 
                    Beleb=cv2.resize(Beleb, (np.shape(img_small)[0] ,np.shape(img_small)[1]), interpolation=cv2.INTER_AREA)
                    
                if largeImg==True:
                    larg=time()
                    img_small2=cv2.resize(img_small, ImgSize, interpolation=cv2.INTER_LINEAR)
                    Beleb2=cv2.resize(Beleb, (np.shape(img_small2)[0] ,np.shape(img_small2)[1]), interpolation=cv2.INTER_LINEAR)
                    print('images upscaled ',time()-larg)
                    numpy_horizontal2 = np.hstack((img_small2, Beleb2))
                    
                
                numpy_horizontal = np.hstack((img_small, Beleb))
                cv2.namedWindow('ItsYou',cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('ItsYou', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                #Text=str(dataN[np.argmin(EuDist)])+ ' EuDist: ' + str(np.argmin(EuDist))
                Text=str(dataN[idx[0].round(2)])+ ' EuDist: ' + str(EuDist[idx[0]])
                
                # FONT_HERSHEY_SIMPLEX        = 0, //!< normal size sans-serif font
                # FONT_HERSHEY_PLAIN          = 1, //!< small size sans-serif font
                # FONT_HERSHEY_DUPLEX         = 2, //!< normal size sans-serif font (more complex than FONT_HERSHEY_SIMPLEX)
                # FONT_HERSHEY_COMPLEX        = 3, //!< normal size serif font
                # FONT_HERSHEY_TRIPLEX        = 4, //!< normal size serif font (more complex than FONT_HERSHEY_COMPLEX)
                font = 2 
                org = (5,17) 
                fontScale = 0.5
                # Blue color in BGR 
                # color = (116, 161, 142) #orig Demo
                color = (0, 0, 1)
                thickness = 1
                numpy_horizontal = cv2.putText(numpy_horizontal, Text, org, font, fontScale, color, thickness, cv2.LINE_AA) 
                
                if largeImg==True:
                    cv2.imshow('ItsYou', numpy_horizontal2)
                #  else:
                cv2.imshow('ItsYou', numpy_horizontal) 
                print('print found image', time()-start6)
                print('-------------------------------------')
                print('time after keypress',time()-start1)
                #print('totaltime ', time()-start1)
                print('-------------------------------------')                
                print('Distance value: ', EuDist[idx[0]].round(2), ' | ' , 'Name: ', str(dataN[idx[0]]),' | ' ,' Filename: ', str(dataF[idx[0]]))
                print('Top five celeb images: ')
                for i in range(5):
                    print(dataN[idx[i]], 'Values: ',EuDist[idx[i]].round(2))
# CLEARING ALL VARIANLES AND CONTINUE WITH THE PROGRAM
                cv2.waitKey(0) #hit any key
                faces_detected=None
                mittleres_Gesicht_X=None        
                img=None
                img_small=None
                pixels=None
                samples=None
                EMBEDDINGS=None          
                cv2.destroyWindow('ItsYou')
                if key == 27: #Esc key
                    break


            else: 
                rame= cv2.putText(frame, 'FACE MUST BE IN FRAME', (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (129, 173, 181), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(900)
                
        else:
            print('noface detected')