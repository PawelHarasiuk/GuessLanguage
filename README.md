##Language Prediction Project
This project is a language prediction application that uses a machine learning model to predict the language of a given text.
It is built using Python and utilizes the scikit-learn library for training and evaluating the model.
The application provides a user interface built with the Tkinter library for inputting text and displaying predictions.

#Getting Started
To use this project, follow the instructions below:

#Prerequisites
Python 3.x
Tkinter
scikit-learn
matplotlib

#Usage
1. Enter the test text in the provided text box.
2. Select the language of the text from the drop-down menu.
3. Click the Add button to add the text and language as training data.
4. Click the Predict button to predict the language of the entered text.
5. The predicted language will be displayed below the Predict button.
6. To evaluate the model's accuracy, click the Evaluate Model button. The accuracy percentage will be displayed below the button.
7. To rebuild the model using the existing training data, click the Rebuild Model button. The model will be retrained, and a confirmation message will be displayed.
8. To load the training data into a table for viewing, click the Load Data button. The training data will be displayed in a table within the application.
9. To visualize the weights of the model for each letter in each language, click the Visualize Data button.
   Separate plots will be shown for each language, illustrating the weights assigned to each letter.

#Data
The training data used for this project is stored in the data/training directory.
Each language has its own subdirectory, and each text file within the subdirectory represents a training sample.
Similarly, the test data is stored in the data/test directory, following the same structure.

#Model
The machine learning model used for language prediction is a Multilayer Perceptron (MLP) classifier.
The model is trained on the provided training data, and the trained model is saved to a SQLite database for future use. 
If no pre-trained model is found in the database, a new model is created and trained.
