import cv2
import os
import numpy as np

def train_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Create the face recognizer
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Load face detection classifier

    faces, ids = [], []  # Initialize lists for storing faces and their IDs

    # Make sure the TrainingImages folder exists and contains the images
    if not os.path.exists('TrainingImages'):
        print("No images found in TrainingImages folder.")
        return

    # Loop through each image in the TrainingImages folder
    for imagePath in [os.path.join('TrainingImages', f) for f in os.listdir('TrainingImages')]:
        img = cv2.imread(imagePath, 0)  # Load the image in grayscale
        id_ = int(os.path.split(imagePath)[-1].split(".")[1])  # Extract the ID from the filename
        faces.append(img)  # Add the image to the faces list
        ids.append(id_)  # Add the corresponding ID to the ids list

    # Train the face recognizer
    recognizer.train(faces, np.array(ids))

    # Create the directory for saving the trained model
    if not os.path.exists('TrainedModel'):
        os.makedirs('TrainedModel')

    # Save the trained model as 'trainer.yml'
    recognizer.save('TrainedModel/trainer.yml')
    print("Training complete. Model saved as 'trainer.yml'.")

# Call the function to train the model and save the 'trainer.yml' file
train_images()
