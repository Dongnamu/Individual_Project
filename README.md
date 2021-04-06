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
  <img src="https://github.com/Dongnamu/Individual_Project/blob/master/images/Training_Accuracy.png" height="150" title="Training Accuracy">
  <img src="https://github.com/Dongnamu/Individual_Project/blob/master/images/Training_Loss.png" height="150" title="Training Loss">
  <br>
  <sup> Convolutional Neural Network training accuracy and loss graphs </sup>
</p>
