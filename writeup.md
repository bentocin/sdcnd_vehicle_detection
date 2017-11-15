# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/overview_class_images.png
[image2]: ./output_images/hog_features_rgb.png
[image3]: ./output_images/hog_features_hsv.png
[image4]: ./output_images/hog_features_hls.png
[image5]: ./output_images/hog_features_ycrcb.png
[image6]: ./output_images/sliding_window_48.png
[image7]: ./output_images/sliding_window_64.png
[image8]: ./output_images/sliding_window_128.png
[image9]: ./output_images/sliding_window_160.png
[image10]: ./output_images/detection_pipeline_1.png
[image11]: ./output_images/detection_pipeline_2.png
[image12]: ./output_images/detection_pipeline_3.png
[video1]: ./project_result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images of the provided dataset. This step was generalized so including additional data, e.g. the [Udacity labeled datasets](https://github.com/udacity/self-driving-car/tree/master/annotations) can easily be done. The get_data() function in "Step 0: Preparation" contains paths to sample images or .csv files describing the dataset and orchestrates the composition of different datasets. The result is a pandas DataFrame that has consistent information about the dataset and can be used later to read in the data.

Here is an overview of randomly picked images of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

The "Step 1: Feature Extraction" sction of the IPython notebook first declares the necessary functions for feature extraction and to build classifiers. Right after that there is a single cell that sets the parameters for the feature extraction. To have consistent settings for the feature extraction for training and later for the video pipeline the feature functions are set up as lambda functions with the settings of the parameter cell and a single image as input. The created pipeline will be used by the `extract_features()` function that loops over each feature extractor in the pipeline and puts together the feature vector. To be able to work with even large datasets, the data read-in process is done with a generator that feeds batches of the dataset through the feature pipeline and only stores the feature vectors instead of the whole image information.

For the HOG feature extraction the function `hog_features()` is used which basically abstracts skimage's hog() function to make working with different channel combinations easier.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.


#### 2. Explain how you settled on your final choice of HOG parameters.

I did not soley focus on the HOG features but tested several combinations of features with a linear SVM as benchmark classifier. The primary goal was to find a parameter settings that reach a solid level of accuracy. The secondary goal was to keep the amount of features low to not lose speed. The final setting I stuck with is HLS colorspace, 9 orientations, 8 pixels per cell and 2 cells per block. This lead to reasonable results with high validation accuracy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

This is done in the "Step 1: Feature Extraction" section. As described above, at first the parameters for the feature extraction are set and afterwards the pipeline is initialized along with sklearn's `LabelBinarizer()` and the data generator. The `get_features_labels()` function is called which reads in image data batch after batch, pushes it through the feature extraction pipeline and returns a set list of feature vectors and binarized labels. This dataset is then split into training, validation, and test sets. Afterwards, a normalizer is fit on the training set using sklearn's `StandardScaler()`. A linear support vector machine is created as classifier and best parameters are chosen among a small variety using sklearn's `GridSearchCV()`. For comparison also some other classifiers are created although for the purpose of this project I stuck with the linear SVM classifier. The linear SVM achieved an accuracy of 99.4% on the validation set and outperformed the other tested classifiers not only in accuracy but also in speed. Because the data was taken from a video stream, the training and validation data might be very similar and lead to overfitting. At least this seems to be the reason for the false positive predictions later on the video pipeline which does not suit a 99% accuracy.

Here are some examples of hog features:
#### RGB
![alt text][image2]
#### HSV
![alt text][image3]
#### HLS
![alt text][image4]
#### YCrCb
![alt text][image5]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In "Step 3: Object Detection" I first declare functions that are later used to perform sliding window search, e.g. `slide_window()` and `find_cars()`. The former one has been used to manually test different settings of overlap, window size and start stop positions. This helped getting an idea of what scale might be relevant for what areas in the image. These findings have then been applied in the `find_cars()` function that adapts the function for hog sub-sampling provided in the course. It takes the feature extraction functions as its input to extract features of the same shape the classifier has been trained on. It uses a scale parameter to have control the size of the sliding window and it takes a parameter to control the overlap of windows. Several settings with scale, overlap and regions of interest have been executed which is how I came to the final settings. The biggest advancement was made using not just one scale but looping over a set of scales from 0.75 to 2.5 with specific regions of interest for each scale. Smaller scales focus on areas of the image that show cars farther away and high scales focus on clos objects. 

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. To get rid of false positives I used the heatmap approach. The windows that predicted a car are all collected and pixels within those boxes are added a heat value of one. This leads to higher values the more windows were detected on that object. 

 Here are some example images:

![alt text][image10]
![alt text][image11]
![alt text][image12]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In the "Step 3: Object Detection" section I implemented heatmap functions to filter out false positives. I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
I improved the false positive detection further by summing the heatmaps over the past six frames and taking this as input for `label()`. This means previous positive detections are prioritized as they stay visible over more frames compared to noise which is popping up randomly.

![video][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

A lot of things were approached with an estimated guess and then tested. Through iterations of trial and error the parameters for feature extraction and the object detection were explored and validated. At the beginning I jumped ahead of myself and tried to generalize the data read in process which might help for future exploration but was not necessary for that project. The testing of different parameter settings was time consuming as the process of feature extraction and classifier training takes some time. This might to some extend be due to my approach to read in the data with a generator function. However, the data also seems to be rather big as the feature vector for an image has almost the size of the image data itself and has data type float.

The pipeline won't be able to track cars that are previously detected and for a period of time occluded ba another car. It also has difficulties when cars are to close together, which might be difficult to use in urban areas. It is also far from real time applicable at the moment as the processing of a single frame takes a little less than a second.

As a next step I would try to get a classifier that generalizes better and experiment with different kinds of neural nets, e.g. a CNN to directly fead images into or a vanilla neural net to classify on the feature vectors. Then I would try to speed up the pipeline to detect objects within a fraction of a second.