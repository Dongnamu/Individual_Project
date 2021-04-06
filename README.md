# A Multi-Target Prototype Exercise Tracker for the Gym Environment
## Introduction
Aim of this project is to propose a new concept of exercise tracking by using an off-the-shelf camera to track individuals, recognise exercises and count repetitions in the gym environment.

## Methods
### Person Identification
Person identification is done by analysing colour histograms. This method showed almost 100% accuracy when only two people are present and wearing clothes that vary widely in colour.

<p align="center">
  <img src="https://github.com/Dongnamu/Individual_Project/blob/master/HSV_view1.png" height="150" title="HSV View 1">
  <img src="https://github.com/Dongnamu/Individual_Project/blob/master/HSV_histogram1.png" height="150" title="HSV Color Histogram 1">
  <br>
  <sup> HSV view and colour histogram of person 1 </sup>
  <br>
  <img src="https://github.com/Dongnamu/Individual_Project/blob/master/HSV_view2.png" height="150" title="HSV View 2">
  <img src="https://github.com/Dongnamu/Individual_Project/blob/master/HSV_histogram2.png" height="150" title="HSV Color Histogram 2">
  <br>
  <sup> HSV view and colour histogram of person 2 </sup>
</p>

### Exercise Recognition
Exercise Recognition is developed with a deep learning approach because deep learning gives a state of the art performance on recognition and classification. CNN model is created and trained in this project, showing over 80% accuracy on the exercise classification.

<p align="center">
  <img src="https://github.com/Dongnamu/Individual_Project/blob/master/images/post-openpose.png" height="150" title="Skeleton">
  <br>
  <sup> Skeleton extraction from an image</sup>
  <br>
  <img src="https://github.com/Dongnamu/Individual_Project/blob/master/images/CNN_architecture.png" height="150" title="CNN architecture">
  <br>
  <sup> Convolutional Neural Network architecture </sup>
  <br>
  <img src="https://github.com/Dongnamu/Individual_Project/blob/master/images/Training_Accuracy.png" height="150" title="Training Accuracy">
  <img src="https://github.com/Dongnamu/Individual_Project/blob/master/images/Training_Loss.png" height="150" title="Training Loss">
  <br>
  <sup> Convolutional Neural Network training accuracy and loss graphs </sup>
  <br>
  <img src="https://github.com/Dongnamu/Individual_Project/blob/master/images/Test_Accuracy.png" height="150" title="Test Accuracy">
  <img src="https://github.com/Dongnamu/Individual_Project/blob/master/images/Test_Loss.png" height="150" title="Test Loss">
  <br>
  <sup> Convolutional Neural Network validation accuracy and loss graphs </sup>
</p>

### Repetition Counter
Repetition counter is created using Principal Component Analysis, which helps to visualise cycle variations of body movements. In this project, said approach showed over 75% accuracy on average.

<p align="center">
  <img src="https://github.com/Dongnamu/Individual_Project/blob/master/images/Chest_PCA_Marked.png" height="150" title="PCA">
  <br>
  <sup> Principle Component Analysis for chests exercise </sup>
</p>

## Evaluation
### Person identification
Colour analysis approach for the person identification in this project is a very simplified approach. This approach fails when people wear the same coloured clothes. By developing the person identification to check more than one feature for each person, it will become more useful in the gym environment. Additionally, the current approach takes a very long time to process. Therefore, pipelining the computation process of the person identification will dramatically increase the speed.

### Exercise Recognition
Exercise recognition in this project showed a respectable performance. However, the accuracy was lower with some exercises compared to others. This is because the number of feature points that contribute to the estimate of the movement of the body varies on exercises. Some exercises mostly move less than 4 feature points, and missing one of the feature points can decrease the prediction accuracy dramatically. To solve this issue, more data sets need to be collected to train the CNN model of Exercise recognition to increase the accuracy of the Exercise Recognition.

### Repetition Counter
Repetition Counter with Principal Component Analysis (PCA) approach has multiple limitations in this project. PCA is a linear dimensionality reduction method, and it is very prone to noisy data. This means it is very sensitive to any gross body movements and poses variations relative to the camera. This causes Repetition Counter to produce unreliable results. Therefore a new approach needs to be considered for the Repetition Counter. One possible approach is to extract feature points on the gym equipment and correlate with feature points from a person to characterise the periodicity of body movements. Then the trained neural network can be used to predict the repetitions accurately. 
