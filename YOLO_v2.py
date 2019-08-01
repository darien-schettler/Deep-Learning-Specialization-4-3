import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
import time
"""

YOLO v2

You will learn to:

1. Use object detection on a car detection dataset
2. Deal with bounding boxes
3. Use the YOLO algorithm

Now we import the libraries and functions we need   { DONE ABOVE }

Important Note: 
As you can see, we import Keras's backend as K
This means that to use a Keras function in this program, you will need to write: K.function(...)

"""

'''

---------------------------------------------------------------------------------------------------------------------
                                            1 - Problem Statement
---------------------------------------------------------------------------------------------------------------------
You are working on a self-driving car project
As a critical component of this project, you'd like to first build a car detection system
To collect data, you've mounted a camera to the front of the car
---> this takes pictures of the road ahead every few seconds while you drive around
---------------------------------------------------------------------------------------------------------------------

CHECK OUT THE VIDEO HERE --> nb_images/road_video_compressed2.mp4 --> TO SEE WHAT WE'RE WORKING WITH

You've gathered all images into a folder and have labelled them by drawing bounding boxes around every car you find
Here's an example of what your bounding boxes look like (---> nb_images/driveai.png)

'''

# # Display the typical bounding box image and how it is classified
# plt.imshow(plt.imread("nb_images/box_label.png"))
# plt.show()

'''
If you have 80 classes that you want YOLO to recognize you can represent the class label  c  either as:
    1. An integer from 1 to 80
    2. An 80-dimensional vector (with 80 numbers) one component of which is 1 and the rest of which are 0
    
The video lectures had used the latter representation
In this notebook, we will use both representations, depending on which is more convenient for a particular step

In this exercise, you will learn how YOLO works, then apply it to car detection

Because the YOLO model is very computationally expensive to train, we will load pre-trained weights for you to use
'''

'''

---------------------------------------------------------------------------------------------------------------------
                                                  2 - YOLO
---------------------------------------------------------------------------------------------------------------------
YOLO (you only look once) is a popular algorithm as it achieves high accuracy while also being able to run in real-time

The YOLO term refers to: 
----> the fact that the algorithm only requires one forward propagation pass through the network to make predictions

After non-max suppression, it then outputs recognized objects together with the bounding boxes
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
                                             2.1 - Model details
---------------------------------------------------------------------------------------------------------------------
First things to know:

The input is a batch of images of shape (m, 608, 608, 3)
The output is a list of bounding boxes along with the recognized classes
---> Each bounding box is represented by 6 numbers (pc,bx,by,bh,bw,c) as explained above
---> If you expand  c  into an 80-dimensional vector, each bounding box is then represented by 85 numbers

We will use 5 anchor boxes

    So you can think of the YOLO architecture as the following: 

            IMAGE (m, 608, 608, 3) ---> DEEP CNN ---> ENCODING (m, 19, 19, 5, 85)

Lets look in greater detail at what this encoding represents....
---------------------------------------------------------------------------------------------------------------------

MAKE SURE TO MAXIMIZE THE IMAGE TO GET A BETTER VIEW
'''

# # Display the typical encoding architecture for YOLO
# plt.imshow(plt.imread("nb_images/architecture.png"))
# plt.show()

'''

If the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object

Since we are using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes
---> Anchor boxes are defined only by their width and height

For simplicity, we will flatten the last two last dimensions of the shape (19, 19, 5, 85) encoding
---> So the output of the Deep CNN is (19, 19, 425)

As shown below....
'''

# # Display the flattening of the last two dimensions
# plt.imshow(plt.imread("nb_images/flatten.png"))
# plt.show()

'''
For each box (of each cell) we compute an elementwise product and extract a prob. that the box contains a certain class

SHOWN BELOW
'''

# # Display the flattening of the last two dimensions
# plt.imshow(plt.imread("nb_images/probability_extraction.png"))
# plt.show()

'''
NO MORE IMAGES WILL BE DISPLAYED THIS TAKES TOO LONG AND I'M GOING TO COMMENT OUT THE PREVIOUS DISPLAYS
'''


# GRADED FUNCTION: yolo_filter_boxes

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """

    # Step 1: Compute box scores

    box_scores = box_confidence * box_class_probs

    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score

    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1, keepdims=False)

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)

    filtering_mask = box_class_scores >= threshold

    # Step 4: Apply the mask to scores, boxes and classes

    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


with tf.Session() as test_a:
    box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed=1)
    boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed=1)
    box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.5)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.shape))
    print("boxes.shape = " + str(boxes.shape))
    print("classes.shape = " + str(classes.shape))


# INTERSECTION OVER UNION --> iou
def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.

    xi1 = max(box1[0], box2[0])  # The upper left corner that is most right/down (X-VALUE) --> (TOP LEFT)
    yi1 = max(box1[1], box2[1])  # The upper left corner that is most right/down (Y-VALUE) --> (TOP LEFT)

    xi2 = min(box1[2], box2[2])  # The lower right corner that is most left/up (X-VALUE) --> (BOTTOM RIGHT)
    yi2 = min(box1[3], box2[3])  # The lower right corner that is most left/up (Y-VALUE) --> (BOTTOM RIGHT)

    inter_area = max((xi2 - xi1), 0) * max((yi2 - yi1), 0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)

    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
    union_area = (box1_area + box2_area) - inter_area

    # compute the IoU

    iou = inter_area / union_area

    return iou


box1 = (2, 1, 4, 3)
box2 = (1, 2, 3, 4)
print("iou = " + str(iou(box1, box2)))


# GRADED FUNCTION: yolo_non_max_suppression

def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initialize variable max_boxes_tensor

    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

    # Use K.gather() to select only nms_indices from scores, boxes and classes
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes


with tf.Session() as test_b:
    scores = tf.random_normal([54, ], mean=1, stddev=4, seed=1)
    boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed=1)
    classes = tf.random_normal([54, ], mean=1, stddev=4, seed=1)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))


# GRADED FUNCTION: yolo_eval

def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """

    # Retrieve outputs of the YOLO model (≈1 line)
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=score_threshold)

    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes


with tf.Session() as test_b:
    yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed=1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
                    tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1))
    scores, boxes, classes = yolo_eval(yolo_outputs)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))

'''
SUMMARY FOR YOLO

1. Input image (608, 608, 3)

2. The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output

3. After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
    Each cell in a 19x19 grid over the input image gives 425 numbers
    425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes
    85 = 5 + 80 where 5 is because  (pc,bx,by,bh,bw)  has 5 numbers, & 80 is the number of classes we'd like to detect

4. You then select only few boxes based on:
    Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
    Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes

5. This gives you YOLO's final output

'''

# You are going to use a pretrained model and test it on the car detection dataset
# As usual, you start by creating a session to start your graph.
sess = K.get_session()

# Recall that we are trying to detect 80 classes, and are using 5 anchor boxes
# We have gathered the information about the 80 classes and 5 boxes in two files "coco_classes.txt" & "yolo_anchors.txt"
# Let's load these quantities into the model by running the next cell
# The car detection dataset has 720x1280 images, which we've pre-processed into 608x608 images
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)

# Training a YOLO model takes a long time & requires a large dataset of labelled bounding boxes for all target classes

# You are going to load an existing pretrained Keras YOLO model stored in "yolo.h5"
#   -- (These weights come from the official YOLO website, and were converted using a function written by Allan Zelener)
#       -- References are at the end of this notebook
#           -- Technically, these are the parameters from the "YOLOv2" model
#           -- We will more simply refer to it as "YOLO" in this notebook

# Run the code below to load the data-set from file
yolo_model = load_model("model_data/yolo.h5")

# This loads the weights of a trained YOLO model. Here is a summary of the layers your model contains
yolo_model.summary()

# The output of the yolo_model is a (m, 19, 19, 5, 85) tensor
# This tensor needs to pass through non-trivial processing and conversion... the following code does that...
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

# yolo_outputs gave you all the predicted boxes of yolo_model in the correct format
# You're now ready to perform filtering and select only the best boxes

# Lets now call yolo_eval, which you had previously implemented, to do this
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)


# Now we run the graph on the image..
# 1. yolo_model.input is given to yolo_model
#       -- The model is used to compute the output yolo_model.output

# 2. yolo_model.output is processed by yolo_head
#       -- It gives you yolo_outputs

# 3. yolo_outputs goes through a filtering function, yolo_eval
#       -- It outputs your predictions: scores, boxes, classes

def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.

    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.

    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes

    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes.
    """

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    plt.show()
    return out_scores, out_boxes, out_classes


for i in range(1, 10):
    out_scores, out_boxes, out_classes = predict(sess, ("000" + str(i) + ".jpg"))
for i in range(10, 100):
    out_scores, out_boxes, out_classes = predict(sess, ("00" + str(i) + ".jpg"))
for i in range(100, 121):
    out_scores, out_boxes, out_classes = predict(sess, ("0" + str(i) + ".jpg"))


'''

What you should remember:

YOLO is a state-of-the-art object detection model that is fast and accurate
It runs an input image through a CNN which outputs a 19x19x5x85 dimensional volume.
The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
You filter through all the boxes using non-max suppression. Specifically:
Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
Intersection over Union (IoU) thresholding to eliminate overlapping boxes
Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well...
    as lot of computation, we used previously trained model parameters in this exercise

If you wish, you can also try fine-tuning the YOLO model with your own dataset

References: 

The ideas presented in this notebook came primarily from the two YOLO papers
The implementation here also took significant inspiration & used many components from Allan Zelener's github repository
The pretrained weights used in this exercise came from the official YOLO website

Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - YOLO: Unified, Real-Time Object Detection (2015)
Joseph Redmon, Ali Farhadi - YOLO9000: Better, Faster, Stronger (2016)
Allan Zelener - YAD2K: Yet Another Darknet 2 Keras
The official YOLO website (https://pjreddie.com/darknet/yolo/)

'''