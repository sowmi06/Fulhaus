# Image Classification 

- Building Classification Model
  - In this project an image classification model has been developed to classify the images based on three classes - "Bed", "Chair", "Sofa". The dataset given is a balanced dataset constitute of 300 images in total with 100 images per class. 
  - An Explorator data Analysis is done; the given images are in different dimensions ranging from (299, 600, 3)  to (5209, 5209, 3) so the images resized to (400, 400, 3),  rescaled the pixel values to 1/255.0 and the image augumentation like rotation, sheering and flipping is done to show minor modification on the images when performing classification. 
  - The final model uses a split of 60% train data, 20% validation data and 20% test data. 5-fold cross validation is done on the model to validate the model.
  - A deep learning classification model is build using the InceptionV3 using the pretrained 'imageNet' weights. The final model is evaluated using the test accuracy score of 93.58%, Precision score of 0.9331, Recall score of 0.9303 and F1score of 0.9310.
  - Saved the model weights in ".h5" file.

- Build an API to access the model
  - Used FastAPI to access the model using the ".h5" file
  - Implemented predictions for the gievn images 

- Created a Docker image of code
- Implement CI/CD pipeline on Github Actions.


## Operating Instructions
CI / CD pipeline - on each push GIT hub action builds docker image and pushes to docker HUB 

- Docker Pull 
        
      docker pull sowmidevaraj/fulhaus:latest
        
- Docker Run 
        
      docker run -p 80:80 sowmidevaraj/fulhaus:latest

 - Access API via postman or browser and upload any images of Bed,Chair,Sofa, api will response the type of image
 
       http://localhost/docs
