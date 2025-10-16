# BrainGPT-Forcasting-future-brain-states-using-Transformer

## Code Description - Brain State Prediction using DSM Method:
The code in this repository is used to train various models for predicting brain signals using the DSM (Dynamic State Modeling) approach.

## Repository Structure:
The repository contains four main folders, each corresponding to a different brain network.
Each folder includes several subfolders, where each subfolder is responsible for predicting a future time sample (t+n) and generating different plots of the prediction results.

• Training Code: Trains the model to predict the corresponding time sample (t+n).
• Loss Plotting Code: Displays the training loss graph for the trained model.
• Final Model: Saved in the corresponding folder after training is completed.

In addition, there is a dedicated data preprocessing script used to prepare the data for training.
Hardware and Software Requirements:
• The code is optimized for GPU execution, and was developed and tested on a high-performance computer located at our college.

## Data Requirements:
• To use the code, you must download the dataset from the HCP (Human Connectome Project) - fMRI brain scans recorded while participants watched movies, with a 7T resolution.

## Note:
The file paths in the code are configured according to our folder structure.
If you download and run the code on your own system, make sure to update the paths to match your local directory structure.
