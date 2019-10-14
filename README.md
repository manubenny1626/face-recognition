# face-recognition
Face Recognition is a recognition technique used to detect faces of individuals whose images saved in the data set. Despite the point that other methods of identification can be more accurate, face recognition has always remained a significant focus of research because of its non-meddling nature and because it is people’s facile method of personal identification.
Categorical crossentropy was the loss function used.Categorical crossentropy will compare the distribution of the predictions (the activations in the output layer, one for each class) with the true distribution, where the probability of the true class is set to 1 and 0 for the other classes. To put it in a different way, the true class is represented as a one-hot encoded vector, and the closer the model’s outputs are to that vector, the lower the loss.
The accuracy is very high as transfer learning was used in it. We have used vgg16 architecture for the transfer learning. Adam is the weight optimiser to be used for the model building.

google drive link:https://drive.google.com/open?id=1nit7sNET1IGRuSTsEZL-YVYdPf8_2LrR

The provided google drive link has the build model and images to be tested.

Image named manu is my image who has created the model and the rest are of my friends.

tranfer_learning.ipynb has the code to test the image.

prediction model useshaarcascade_frontalface_default.xml for the face detection.

only the face in the image is loaded into the model and the model perdicts face in the image. 

training

the model was trained on 4986 images for our image classifier, we only worked with 6 classifications so using transfer learning on those images did not take too long.
Depending on your image size we found best that 224, 224 works best. Then we created a bottleneck file system. This will be used to convert all image pixels in to their number (numpy array) correspondent and store it in our storage system. Once we run this, it will take from half hours to several hours depending on the numbers of classifications and how many images per classifications. Then we simply tell our program where each images are located in our storage so the machine knows where is what. Finally, we define the epoch and batch sizes for our machine. For neural networks, this is a key step. We found that this set of pairing was optimal for our machine learning models but again, depending on the number of images that needs to be adjusted.

we did image augmentation such as image rotation, transformation, reflection and distortion.
Once the files have been converted and saved to the bottleneck file, we load them and prepare them for our convolutional neural network. This is also a good way to make sure all your data have been loaded into bottleneck file. this was repead for validation and testing set as well.

we flatten our data. since this is a labeled categorical classification, the final activation must always be softmax. It is also best for loss to be categorical crossenthropy but everything else in model.compile can be changed. Then after we have created and compiled our model, we fit our training and validation data to it with the specifications we mentioned earlier. Finally, we create an evaluation step, to check for the accuracy of our model training set versus validation set.

Then the model is saved. The model is saved as face_rect.h5 and has attached to google drive link.

dependencies

keras
tensorflow
numpy
glob
mathplotlib
cv2
pillow
sklearn
base64
io
json
radom


haarcascade xml file ca be downloaded from the google drive. 
https://drive.google.com/open?id=1nit7sNET1IGRuSTsEZL-YVYdPf8_2LrR
