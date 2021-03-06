# CelebFaceMatcher


![alt text](https://github.com/JanderHungrige/CelebFaceMatcher/blob/master/jandemo.png?raw=true)

# Introduction 

The demo will take a video stream from your webcam, or embedded camera, and detects all faces in it. If a face is found and marked, you can hit space to indentify which celebrity looks the marked face most closely. The detected image is croped and analysed with a Neural Network. The demo will display the resulting celebrity with his image and name next to your face. To restart the stream hit space again or cancel with ESC.

With this repo, you will find all the steps to create a celebrity- face comparison demo. **A detailed describtion can be found [in this blog](https://janwerth.medium.com/1e4e9de660cc?source=friends_link&sk=c938b9ebfd55f8dec0b486ca746df763)**

A version that runs more smooth on embedded hardware, can be found [here](https://github.com/phytec/demo-celebrity-face-match)

## Prerequisite

* python 3.6+ environment (I recommend [Anaconda](https://www.anaconda.com/) using [virtual environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)),
* [icrawler](https://pypi.org/project/icrawler/) , 
* [TensorFlow 2.x](https://pypi.org/project/tensorflow/),
* [tflite_runtime](https://pypi.org/project/tflite/),
* [pandas](https://pypi.org/project/pandas/),
* [numpy](https://pypi.org/project/numpy/),
* [matplotlib](https://pypi.org/project/matplotlib/), 
* [scipy](https://pypi.org/project/scipy/), 
* [opencv-python](https://pypi.org/project/opencv-python/),
* [and the tf.keras-vggface model](https://github.com/JanderHungrige/tf.keras-vggface).

The tf.keras-vggface model you need: https://github.com/JanderHungrige/tf.keras-vggface

To install the tflite_runtime, download and install [this x86 wheel file](https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp36-cp36m-linux_x86_64.whl) and install via pip install path_to_file if the above (ARM) does not work.

## The model and files 
If you are the fast type, you can download the tflite model and dataset here:
```
wget ftp://ftp.phytec.de/pub/Software/Linux/Applications/demo-celebrity-face-match-data-1.0.tar.gz
tar -xzf demo-celebrity-face-match-data-1.0.tar.gz
```

If you use this repo and downloaded the model, please save the model into the model folder or adjust the path in the 3-Laeuftauf8Plus_v4.py file. 

## The Files explained
Otherwise, use the files as following to create the demo from scratch or create your own demo in an adapted way.

* **1a-Image-Crawler.py** to crawl for celebrity images or the images you want to use as base. 
* **1b-get_faces_and_crop.py** to only get the faces from the freshly downloaded images and cut them into size 224x224 (the size can of course be altered, but the included model will only function porperly with 224x224)
* **1c-quant_direct_from_model.py** to quantize your model to be able to use it on a embedded NPU
* **1e-proof_images_analysis.py** if you want to use license free files, but you get a lot of rubbish with the crawler. Here the images will be compared within each folder to find outliers, or against a gold standart Embeddings file from good(non creative commons) images. 
  * If you do not have this gold standart Embeddings file, see next point. If you want to plot your embeddings per folder use **1d-proof_images_plotting.py**
* **2 -Create embeddings database.py** to create Embeddings for the crawled and cropped faces. 
  * This can also be used to create a Embeddings_file to be used as the gold standart with non-creative commons files (which are moslty better, but cannot be used commercially). This non creative commons images will only be used to compare your creative commons images and find/delete missmatches. To get the non creative commons images, use the **1a-Image-Crawler.py** again without the filter: *license= (commercial, reuse)*.
* **3-Laeuftauf8Plus_v4.py** : Now that you have your qunatized model, an Embedding file from images and the image, this file will run the demo


Have fun

## License
This project is licensed under the Apache License Version 2.0. See the file *LICENSE* for detailed information.
