This project aims to improve face detection accuracy of deep neural networks on abstract art, such as Picasso's cubic paintings.

**Read the project report [here](https://github.com/bcliu/picasso-research/blob/master/picasso-project-report.pdf).**

This project depends on Caffe (http://caffe.berkeleyvision.org/). To use the Matlab version of the code, run "make all matcaffe" in Caffe root; to use the Python version, run "make all pycaffe" in Caffe root.

Support vector machine code depends on scikit-learn library (http://scikit-learn.org/stable/index.html).

1. constants.py contains root of Caffe and images and source code of the project. Change the values to correct paths.

2. shrink-images.py: Shrink all images in a directory to smaller images with white padding
    * First argument: path to directory
    * fraction argument: how many times smaller the new image should be

3. svm.py: Support vector machine code to classify responses of AlexNet. Uses fc6 layer response of all images to learn a classifier that distinguishes the three types of images:
    * type1 argument: path to directory containing images of type 1
    * type2 argument: path to directory containing images of type 2
    * others argument: path to directory containing images of all other types

4. svm-2-classes.py: Train an SVM classifier with two classes (e.g. eye images, and other images that are not eyes),
                  by taking a specified layer from AlexNet response.
