# CelebFaceMatcher


![alt text](https://github.com/JanderHungrige/CelebFaceMatcher/blob/master/jandemo.png?raw=true)

With this repo, you will find all the steps to create a celebrity- face comparison demo. A detailed describtion can be found [in this blog](https://janwerth.medium.com/1e4e9de660cc?source=friends_link&sk=c938b9ebfd55f8dec0b486ca746df763)

If you are the fast type, you can download the tflite model and dataset [here](ftp://ftp.phytec.de/pub/Software/Linux/Applications/demo-celebrity-face-match-data-1.0.tar.gz) ftp://ftp.phytec.de/pub/Software/Linux/Applications/demo-celebrity-face-match-data-1.0.tar.gz

If you use this repo and downloaded the model, please save the model into the model folder or adjust the path in the 3-Laeuftauf8Plus_v4.py file. 

Otherwise, use the files as following to create the demo from scratch or create your own demo in an adapted way.

* use *1a-Image-Crawler.py* to crawl for celebrity images or the images you want to use as base. 
* use  *1b-get_faces_and_crop.py* to only get the faces from the freshly downloaded images and cut them into size 224x224 (the size can of course be altered, but the included model will only function porperly with 224x224)
* use *1c-quant_direct_from_model.py* to quantize your model to be able to use it on a embedded NPU
* use *1e-proof_images_analysis.py* if you want to use license free files, but you get a lot of rubbish with the crawler. Here the images will be compared within each folder to find outliers, or against a gold standart Embeddings file from good(non creative commons) images. 
  * If you do not have this gold standart Embeddings file, see next point. If you want to plot your embeddings per folder use *1d-proof_images_plotting.py*
* Use *2 -Create embeddings database.py * to create Embeddings for the crawled and cropped faces. 
  * This can also be used to create a Embeddings_file to be used as the gold standart with non-creative commons files (which are moslty better, but cannot be used commercially). This non creative commons images will only be used to compare your creative commons images and find/delete missmatches. To get the non creative commons images, use the *1a-Image-Crawler.py* again without the filter: *license= (commercial, reuse)*.
* Now that you have your qunatized model, an Embedding file from images and the image, you can run the demo with *3-Laeuftauf8Plus_v4.py* 

The demo will take an image with your webcam or embedded camera and if a face is found and marked, you can hit space. The detected image is croped and analysed with a Neural Network. The result will be displayed with the image and name of the celebrity. Again, for more details visit the blog mentioned above.  
