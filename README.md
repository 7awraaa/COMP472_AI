# COMP472_AI

**Team OB_18**

Alma Almidany (40197854) - Data Specialist

Carmen Derderian (40244084) - Training Specialist

Hawraa Al-Adilee (40246450) - Evaluation Specialist

**Link to the full dataset**

https://www.kaggle.com/datasets/thienkhonghoc/affectnet

**Link to path files and REPORT PARTS 1-2-3**

The report PDF file and the path files exceed the file size that can be uploaded. It can be found in the following google drive folder.

https://drive.google.com/drive/folders/1OYFo1RMOfkh8MJf-Y-fekALm1wqm0_HF?usp=sharing

**Purpose of each file**

--Part 1--

*class_distribution.py*: This code is used to visualize the images accross different classes in the dataset. It reads a JSON file containing image labels, counts the number of images in each class, and generates a bar graph to display the results. The x-axis represents the classes, and the y-axis represents the number of images in each class.

*label_images.py*: This script sorts images into different folders based on their corresponding emotions and creates a JSON file with the image labels. The images are read from the `data/cleaned/emotion` directory (generated from select_images.py), sorted into their specific emotion folder, and saved in the `data/labeled/emotion` directories. The JSON file, `labels.json`, is used for labeling and to plot the class distribution plotting purposes in other scripts.

*pixel_intensity_distribution.py*: This file is used to compute the pixel distribution of the all photos (.jpg) in each of the four labelled folders by reading the images from the `data/labeled/emotion` directories. The emotion specific images are split into their Red, Green, and Blue channels. After the code completes executing (~15 minutes), a pixel intensity distribution graph is returned with a title corresponding to the folder name: the x-axis represents the pixel intensity and the y-axis represents the the frequency.

*resize_images.py*: This script resizes all images to a consistent dimension of 256x256 pixels for data cleaning. It reads images from the `data/raw/emotion` directories, processes them (including optional brightness adjustments and rotations), and saves the resized images in the `data/cleaned/emotion` directories. It takes in the raw 500 photos of each dataset and applies these modifications ensuring uniformity in the dataset.

*sample_images.py*: This script displays 15 sample images from each class with each image's pixel intensity histogram next to it. It reads images from the `data/labeled/emotion` directories, randomly selects 15 images from each class, and plots them alongside their pixel intensity histograms. 

*select_images*: his script randomly selects 500 images per class from the public AffectNet dataset on Kaggle and stores them in the `data/raw/emotion` directories. It goes through the corresponding subfolders of each emotion and creates a folder labelled with its corresponding folder.

*test_images.py*: This script randomly selects 100 images from each labeled dataset subfolder for testing purposes. It moves the selected images from the `data/labeled` directories to the `data/test` directories, ensuring that they are separate from the training images. The remaining images are moved to the `data/evaluation` directories for further evaluation.

--Part 2--

*python evalution.py*: This script generates metrics for each model and classifies and tests the best model on a randomly chosen image from the dataset. 

*python confusion_matrix.py*: This script generates a detailed confusion matrix about how the images have been classified and the actual class they belong to. **Generates the matrix of one model at a time. Modifications must be done at three lines of the code which are flagged as "TODO"**

*python general_confusion_matrices.py*: This script generates TP, FN, FP, and TN values for each class in each CNN model.  

*main_model.py*: This script trains the main model. It generates a path file once the model is done training (best_facial_expression_model.pth).

*variant1_model.py*: This script contains the first variant of the main model which contains an additional convolutional layer.It generates a path file once the model is done training (path_variant1.pth).  

*variant2_model.py*: This script contains the second variant of the main model which differs by the kernel size. In this model, the kernel size upgrades from 3 to 5, and the padding from 1 to 2. It generates a path file once the model is done training (path_variant2.pth).  

*Plot_Convolutional_Layers.py*: This script generates a graph to visualize the convolutional layers of the main model (best performance). 

--Part 3--

*all_labeled_images.csv*: This file contains all labeled images for the bias analysis on attributes.

*labeling_interface.xml*: This file contains the script used during the bias attributes labeling process with Label Studio.

*Bias - Folder*: This folder contains all labeled images. It stores images per categories. Each category folder contains subfolders, one for each expression.

*Kfold_fixbias.py*: This code is to fix the bias using k folds. Some modification were made to the main model such as, data augmentation, regularization techniques, and balanced sampling to ensure fair and unbiased model performance. The script performs training and evaluation.

*main_model_aug.py*: This file trains the main model with the implemented changes for bias mitigation. It generates a path file once the model is done training (best_facial_expression_model_aug.pth)

*bias_evaluation.py*: This file contains the script used to evaluate the performance of the model with changes for bias mitigation. It used the following path file generated by the file *main_model_aug.py*: best_facial_expression_model_aug.pth. The evaluation process employed is identical to the one of *evalutation.py*. 


*Notes*: Each emotion has a total of 503 images as there is a picture of each teammate for each emotion.

**To execute the code**

1. **Ensure the require packages are installed**

    *pip install numpy matplotlib pillow scikit-learn*

2. **Navigate to the script folder**

    *cd scripts*

3. **Select Images**

    *python select_images.py*

4. **Resize Images**

    *python resize_images.py*

5. **Label Images**

    *python label_images.py*

6. **Visualize Class Distribution**

    *python class_distribution.py*

7. **Pixel Intensity Distribution**

    *python pixel_intensity_distribution.py*

8. **Sample Images**

    *python sample_images.py*

9. **Select Test Images**

    *python test_images.py*

----------- Training, Evaluating and Applying the Models -----------

1. **Training**

    To train the main model: *python main_model.py*
   
    To train model variant 1: *python variant1_model.py*
   
    To train model variant 2: *python variant2_model.py*

2. **Evaluation**

    To generate metrics: *python evalution.py*: 

    To generate detailed confusion matrix: *python confusion_matrix.py*

    To generate TP, FN, FP, and TN values (general confusion matrix): *python general_confusion_matrices.py*

3. **Application**

    To train the models with a new dataset:
    A. Main Model
   
        1. Insert the path to the new dataset on line 39 of *main_model.py*.
        2. Train the model with the following command: *python main_model.py*
   
    B. Model Variant 1
   
        1. Insert the path to the new dataset on line 43 of *variant1_model.py*.
        2. Train the model with the following command: *python variant1_model.py*
   
    A. Model Variant 2
   
        1. Insert the path to the new dataset on line 39 of *variant2_model.py*.
        2. Train the model with the following command: *python variant2_model.py*
   
    To evaluate the trained models:
   
    Follow same steps as in Evaluation (step 2). Make sure to insert the new dataset path on the following lines:
   
    a. Line 35 in *evaluation.py*
    b. Line 326 in *confusion_matrix.py*
    c. Line 336 in *general_confusion_matrices.py*

----------- Training and Evaluating the Model - KFold -----------

**Training and Evaluation**

    1. Insert the path of the dataset on line 30 of *Kfold_fixbias.py*
    2. To train and evaluate the main model with KFold: *python Kfold_fixbias.py*
    - Once the training is over, the following metrics are generated: accuracy, precision, recall, F1 Score, precision (micro), recall (micro), F1 Score (micro), precision (macro), recall (macro), F1 Score (macro)


----------- Training and Evaluating the Model - Bias -----------
1. **Training**

    To train the model with the labeled dataser:
   
        1. Insert the path to the new dataset on line 30 of *main_model_aug.py*.
        2. Train the model with the following command: *python main_model_aug.py*

3. **Evaluation**

    To generate metrics (accuracy, precision, recall, F -Score) for each category:
   
        1. Modify the path to the category to be evaluated on line 30 of *bias_evaluation.py*
        2. Modify the print string accordingly on line 130 of *bias_evaluation.py*
        3. Evaluate the model wotj the following command: *python bias_evaluation.py*
        4. Repeat for all six categories
