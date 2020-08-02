<!--
* Get pandoc working, generating to latex
* Get BibTex working with pandoc, generating to latex
* Add entries to BibTex
* Write Draft Introduction
* Write technical discussion of neural Networks
* Write technical discussion of Convolutional Autoencoders
* Write partial section of methods
* Select ~10k images, preferably of a manageable number of artists, of a given time, with prices
* Do some basic analysis on the artists, periods, and prices of those images
* Train DCEC-paint on those images with variable number of clusters and whatnot
* Make t-SNE diagrams
* DECISION whether you want to pursue prediction

if clustering:
* want to evaluate the performance against traditional methods

elif prediction:
* Rebuild network with fully connected prediction layer
    - Use google cloud storage library to update things
    - Read configurations from environment variables set in the job.yaml
    - Log the data and plot it
    - What is your loss function? - What is their loss function?
    - Get rid of run model.py
    - How do you set this? They did it with Kmeans, what should you do it with?
        + Mean of the training set
        + Median of the training set
        + Something random from the training set?
        + 0?
* Select a meaningful 10-15k photos? With artist? Is that possible?
* What about selecting traditional visual characteristics of the photo, how does that perform?
* Train, tune, validate, test on 50k print/photo dataset
    - clean that dataset??
    - Hopefully no need to clean that dataset further
* Get basic statistics about datasets

IMPROVEMENTS
* See pg 4 Guo et al. - Batch Normalization, LeakyReLU, hypertuning the embedded space dimension

NEED:
* Cross depiction problem - Castellano and Vessio
-->

<!--
Rendering
* Need TOC
-->

# Deep Convolutional Autoencoder Prediction of Art Action Prices

## Abstract
<!-- TODO Write this after you finish the paper -->

## Introduction
<!-- This is where your state the motivation -->
<!-- contributions to the field -->

The widespread digitization of fine art over the past two decades has coincided
with large advances in computational image analysis. As museum and gallery
collections move onto the internet, millions of artworks have been made
available for view at the click of a mouse [cite google thing]. A proliferation of researchers
have sought to analyze these digital collections, contributing methods that
excel in a wide variety of computer vision tasks from classification problems (genre,
style, artist, historical period) [Ceternic, 2018], to visual relationships
between paintings [Garcia, 2019].

The complete visual and emotional effect of a painting is a combination
of many factors, for example the color, texture, spatial complexity, and contrast. An art expert
recognizes those qualities and is able to place a work in its historical
and artistic context. That task, however,
is difficult to articulate and exceedingly difficult to generalize to the set
of all artwork. Early attempts at computational art analysis, nevertheless,
attempted to build a similar model by first engineering and extracting domain specific features
from the pixel space (corners, edges, SIFT), and using those feature vectors
as input to a model. These techniques formed the backbone of early attempts at
object recognition within and outside the art domain, and saw some success in
evaluating art. In recent years, however, the field has undergone large advancements in
computer vision techniques, in particular
Convolution Neural Nets (CNNs) which have demonstrated outstanding results in
extracting semantic meaning from digitized work. These results are impressive
both in relation to earlier, feature engineering based attempts, and
in comparison to the perceived complexity of recognizing the distinct visual
appearance of an artwork.

Rather than using engineered features that attempt to
proxy semantic attributed, CNNs attempt to learn relevant features from a large
set of training images. CNNs applied to fine art related tasks have benefited
from both large annotated sources of art data, as well as enormous datasets of
non-art related images. In the former case, annotated art datasets allow researchers
to train classification models without hand labeling artist and genre metadata.
This has led to a number of successful models that can identify period and even
artist [cite here]. In the latter case, large non-art related image datasets
have been used to pre-train object recognition models for art-related tasks.

The large availability of annotated datasets has led to many authors focusing on
supervised learning tasks. Comparatively little art research focuses on unsupervised
learning, including clustering. Clustering artwork has a number of applications to
aid the art expert in knowledge discovery, including identifying distinct periods within an artists career,
shared techniques between groups of artists, and distinct periods within unattributed
groups of work (e.g. ancient east Asian art).

In this work, we replicate a CNN clustering algorithm, Deep Embedded
Convolutional Clustering (DCEC) first proposed by [Guo et al., 2017] and adapted
by Castellano and Vessio, 2020 to an art specific dataset. DCEC is composed of
two components, a convolutional autoencoder (CAE) and a clustering layer
attached to the CAE. Autoencoders perform non-linear dimension reduction of the
image in two steps: an encoder which develops a mapping from the input image
to a highly reduced latent (embedded) pixel space, and a decoder which maps
from the latent space to a reconstructed image of equivalent dimensions as the
input. By attaching a clustering layer to the CAE, and simultaneously optimizing
for both clustering and image reconstruction loss, DCEC ensures that clustering
of the images occurs within the constraints of a latent space i.e. clusters
are formed in the presence of semantic attributes deemed meaningful by their
inclusion in the latent space.

We evaluate clustering performance against two separate datasets of
digitized artwork, both scraped from Christies auction house's [cite] public
record of auction sales. The first dataset we cluster is a set of [TODO: n (b/w)]
images, and the second is a set of [TODO: n (east/asian)]. The only known prior
work with this algorithm used two datasets, one of a set of paintings by 50
well known artists and the collected works of Pablo Picasso. The images used in
this work include more obscure artists, as well as a higher proportion of
intra-genre work, which exercises the algorithm's performance in face of lower
magnitude differences in the feature space.

TODO Incorporate this sentiment somewhere else - From Xie et al.
<!-- One   branch   of   popular   methods   for   clustering   isk-means  (MacQueen  et  al.,  1967)  and  Gaussian  MixtureModels (GMM) (Bishop, 2006).   These methods are fastand applicable to a wide range of problems. However, theirdistance metrics are limited to the original data space andthey  tend  to  be  ineffective  when  input  dimensionality  ishigh (Steinbach et al., 2004). -->

<!-- 
IDEAS
* clustering artwork is hard - Castellano and Vessio, 2020
    - recognizing meaningful patterns in accordance with domain knowledge and visual perception is hard
    - applying traditional clustering and feature reduction techniques to the high dimensional pixel space is ineffective
* Computer vision as a tool for recognizing patterns in artwork
* See Google's thing about artwork similarity
* Humans perceive meaningful patterns in artwork by recognizing the stylistic characteristics of it
    - color, texture, spatial complexity, etc
* But that is really hard to conceptualize and quantify
* CNNs are good at recognizing visual-related patterns from the low level pixel values
* Computational analysis of art
* Steps in the direction toward generative artwork. Realizing a vision of art created for us

* Much of this work benefits from large annotated sources/labeled data, and has benefited from non-fine art datasets such as imagenet

MOTIVATION
* "It can be used to support art experts in findings trends and influencesamong painting schools, i.e. in performing historical knowledge discovery. Analogously, it can be used to discoverdifferent periods in the production of a same artist. The model may discover which artworks influenced mostly thework of current artists. It may support interactive navigation on online art galleries by finding visually linked artworks,i.e. visual link retrieval. It can help curators in better organizing permanent or temporary expositions in accordancewith their visual similarities rather than historical motivations"
* representation learning vs feature engineering
* unsupervised clustering vs supervised learning -> broadens the potential datasets that we can use
 -->

## Related Work
<!-- DCN efforts, specifically deep clustering, like DEC, DCEC, DCEC-Paint -->
<!-- TODO Efforts to quantify art prices, especially using extracted, not learned features -->

As institutions have digitized their art collections over the preceding decades,
many researchers have applied computational image analysis methods to art domain.
Although applications are quite broad, a large number of efforts have focused
on classification (of artist, genre, period, movement, medium, etc), object recognition,
visual similarity (between artwork), the cross depiction problem (distinguishing
the same type of object in different representations).

Initial attempts to analyze art emphasized feature engineering and extraction
[cite some people] in which domain specific characteristics of artwork are
identified and a given artwork's relative presence or absence of those
features is used as input to a model. For example, (see Garcia)
<!-- TODO: What are some places this was done and what are the features that were extracted -->

While feature engineering has been shown to be effective, it's limited in its
requirement for a comprehensive set of pre-identified features for modeling the
image characteristics. This shortcoming is especially apparent in tasks which
attempt to consider the image as a whole, a task which seems especially complex
given the myriad possible "features" that could be found in an m x n x p
dimensioned image. Over the past decade, the computer
vision community has focused on designing algorithms which, rather than rely on
extracted features, learn a relevant feature set through a training process.
Applying deep learning concepts to art analysis has proved immensely fruitful
in a host of subfields. Cetinic et al., demonstrated the effectiveness of fine
tuned CNNs in classifying artwork by artist, genre, style, time period, and
even national artistic context. blah blah blah

The majority of art analysis research employs supervised learning, in part due
to the availability of numerous and large labeled datasets. In recent years,
however, a few authors have focused on unsupervised learning, in particular
clustering. Outside of the art domain, Xie et al., 2016 proposed a Deep Embedded
Clustering (DEC) algorithm which selects clusters in two steps: first learning a reduced
dimensionality set of features using stacked autoencoders (SAE) and second
using stochastic gradient descent with a soft assignment loss function to perform
clustering. Guo et. al expanded on this work by using Convolutional Autoencoder
rather than a stacked auto encoder, and by jointly optimizing for both clustering
loss and image reconstruction loss so as to avoid corrupting the semantically
meaningful feature space during the clustering component (Deep Convolutional
Embedded Clustering (DCEC)). Finally, Castellano and Vessio, inspired by Guo et
al., tweaked DCEC for the larger and more complex images found in art datasets.

In this work we replicate the algorithm proposed by Castellano and
Vessio, by applying it to different art datasets.


* Cetinic et al, 2018 - performed 5 different classification tasks on 3 large art datasets


## Convolutional Neural Networks
<!-- Motivation for what NNs offer in general -->
<!-- What do NN offer to image problems -->
<!-- What do they offer to this specific problem -->

<!-- 
* What is an image?
    - An image is a set of pixels, each with three dimensions, arrayed in a grid.
    - The given arrangement of these pixels is what gives these images their texture/shape/complexity, etc
 -->

Convolutional neural nets have become a power tool in computer vision over the
previous decade. Their ability to extract meaningful patterns [cite here] from
images has led to their widespread usage in vision problems such as object
detection and facial recognition [cite here]. Autoencoders are a type of unsupervised neural
net which learns a mapping from a high dimensional data space to a lower dimensional
feature space. Like CNNs, Autoencoders have proved wildly useful in the computer vision
domain due the high dimensionality and complexity of images [cite here]. This
section provides a brief and incomprehensive overview of the mathematical
concepts underlying CNNs and Autoencoders. For a gentle introduction to neural
networks and deep learning, see [Michael Nielson]. For a comprehensive look at
neural networks including autoencoders, see [cite textbook].

### Artificial Neural Network Basics
<!-- How and why do NNs work -->

#### Architecture

Artificial neural networks are non-linear functions, F_nn: R^I -> R^K, where I and
K are the dimensionality of the input and output spaces, respectively [cite englebrecht].
Modeled after their biological equivalents, they achieve this non-linear functionality
through composition of layers of artificial neurons where an individual neuron is a nonlinear function,
y = f(x + b), typically y = [0, 1] or y = [-1, 1] , called an activation function, which accepts n > 0 input signals (x) and outputs a single value
as a function of the inputs and the learned weights (w) and biases (b) for each
interneuron connection [diagram here, steal from figure 1.3 in englebrecht]. Thus,
for each edge (connection) between the jth neuron in layer i and the kth neuron in
layer i +1, the network learns a particular weight (w_(i+1)_jk) which represents
the relative importance of that component of the input signal. When
layered together, the output of the neuron in the ith layer is the input to the
neuron(s) in layer i + 1, forming a structure similar to the one depicted in the
figure x [diagram here, steal from figure 1.4 in englebrecht], typically a directed
acyclic graph (DAG) (footnote: the
actual structure of a particular type and instance of network is highly variable
within these general constraints. The activation function, number of inputs, number
of layers, interconnectedness of the layers, and even direction of connections networks
all vary depending on the desire of the practitioner).

<!-- Activation functions -->
In general, an activation function is a monotonically increasing function, F_AN: R -> [0, 1] or F_AN: R -> [-1, 1]
such that:

F_AN(-inf) = 0 or F_AN(-inf) = 1 and  f_AN(inf) = 1

There are many activation forms but we focus on a single, and most common, form,
the sigmoid:

equation 2.11 in englebrecht

This function has highly beneficial properties for learning. It is continuously
differentiable along the real numbers, which, combined with the fact that it is
monotonically increasing means that a gradient can be calculated at any place,
facilitaing "learning" as discussed in the section below. The fact that its derivative
asymptotically aproaches zero for large magnitude input values is detrimental to
quick learning, however, this can be compensated for by tuning specific parameters
in the network.


<!-- Neural networks can be shown to approximate any function (cite this as well) -->

#### Artificial Learning

Neural networks can be shown to approximate any continuous function [TODO cite this, pg. 28 in englebrecht] - (Hornik,  Kurt.    Approximation  capabilities  of  multilayerfeedforward networks.Neural networks, 4(2):251–257,1991.)
to some
desireable level of accuracy. NNs achieve this impressive result by "learning"
the appropriate weights and biases, progressively updating these values until
an approximation is sufficiently close. There are many different learning algorithms,
components for updating the weights and biases. In the interest of brevity, we
focus on the most widely used learning rule, Gradient Descent. For an introduction
to additional rules such as the Widrow-Hoff, Generalized Delta, and Error-Correction
learning rules, see Englebrecht.

As mentioned abouve, there are a number of different types of neural networks.
Below, we will examine in depth the convolutional autoencoder. However, to discuss
Gradient Descent, consider a fully connected 3 layer feed forward neural net,
depicted in figure {blank}. Each neuron in the ith layer is connected to each
neuron in the i + 1 layer and inputs are passed only forward (that is, right to left)
through the network.

#### Stochastic Gradient Descent

Gradient Descent attempts to minimize the value of an error function [footnote:
Also called an optimization or Cost function], Epsilon(y - y'),
by progressively updating the network weights and biases in such a way as to follow
the first derivative gradient of Epsilon with respect to the weights (partial epsilon wrt w). For example,
consider a common form for epsilon, the sum of squared errors:

(Also see section 2 in Nielson)

2.17 in englebrecht

where y is the expected output for an input vector (x), and y' is the actual output.
Considering a single example input, x, the weights are updated according to the following
equation:

w_i(x) = w_i(x - 1) + deltaw_i(x)

where

deltaw_i(x) = eta (- partial Epsilon wrt w_i)

and

see englebrecht equation 2.20

Thus, at each learning step (an epoch), gradient descent updates the weights in the direction
that results in the largest reduction in Epsilon.

##### Backpropagation

From these equations, however, it remains unclear how each weight in the network
is updated. There are a number of detailed proofs for backpropagation, see, for example,
englebrecht, but, the algorithm has two phases:

1.  Inputs are passed through the network and output values calculated, providing
    the values needed to calculate epsilon and the derived gradients
2.  The error signal from the output layer is propagated backwards through the
    network such that each weight and bias is updated according to its effect on
    the overall error of the cost function.

(TODO you need to explain back propagation in more detail here)


### Convolutional Autoencoders

Before delving into the specifics of convolutional autoencoders, consider the
image below.

TODO figure

<!-- TODO this should have a particular image associated with it, something famous -->
While to the human eye an image may appear a mosaic of shapes and colors,
computationally images are matrices of numeric values pertaining to pixel
intensity. In the case of black and white images, there is a single channel,
meaning an image can be represented by a single matrix of size m x n. Color images are
3 dimensional, accounting for the increase from one to p > 1 channels (3, in the case
of RGB images). Consider a neural network tasked with learning relevant features
from an input dataset of x, m x n x p images.

Intuitively, neural networks seem like an appropriate tool for identifying
relationships between pixels. An architecture of many interconnected neurons
seems as though it should have the ability to progressively weight more meaningful
patterns in the data (over the course of many epochs) by progressively weighting
certain neuron relationships.

Convolutional neural nets build on this intuition by attempting to progressively
tease out these patterns through sequential layers in the network.

#### Convolutional Neural Networks

<!-- Cite http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf -->

Convolutional neural networks are a type of deep neural network {footnote: deep
means more than a single hidden layer} with an architecture that is well suited
to image analysis. In particular, convolutional neural nets are better able to
extract the spatial structure of an image than traditional networks because they
limit the connectectedness of the network by ensuring connected neurons correspond
to spatially adjacent input pixels, by sharing weights among edges within the same
layer, and by pooling subsequent layers to provide dimension reduction. For
a full discussion of convolutional neural nets, see
[levun](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf).
We summarize briefly here.

Consider an input image of size 4 x 4 x 1, that is, 16 total pixels, depicted below.

TODO image

In a fully connected network, each pixel value would be input to each neuron in the
first hidden layer. Convolutional neural networks, differ by defining a local receptive
field of size k, where (kxk) is the number of adjacent inputs that will connect to
the jth neuron in layer i.

TODO figure

For each neuron in the hidden layer, this local receptive field is moved adjacently
by a stride length, (l). For example, a stride length of 2 will correspond to a total
of 9 local receptive fields, meaning a total of 9 neurons in the first hidden layer.
By limiting the number of input signals passed to a single neuron, each neuron
receives information only from adjacent pixels, rather than information from
every input in the network, including pixels/neurons that are spatially distant.
Intuitively, this seems like an appropriate way to extract spatial meaning.

Furthermore, by design, each neuron in the ith hidden layer has the same weights
and biases. So, for our example, the 2*2 array of weights and biases input to the
jth neuron in the ith layer are identical to every other array of weights and biases
in the ith layer. By making this restriction, the network ensures that all the neurons
in the ith layer are detecting the same spatial structure. This map from an input
layer to a hidden layer is called a feature map, and the set of weights and biases
that define a feature map is called a filter or a kernal (cite something here).
By increasing the number of feature maps at each layer, the network is able to
detect multiple features (Bengio,  Yoshua,  Courville,  Aaron,  and  Vincent,  Pascal.Representation learning: A review and new perspectives.2013.).

Finally, convolutional NNs pool the activations of the convolution layers in the
aptly named, pooling layers. By pooling adjacent activations, typically by taking
the maximum activation from a lxl sized area, the featue maps undergo
dimension reduction. Intuitively, this can be thought of as a function which outputs
whether a certain feature is found anywhere within a subsection of a layer.

The combined effect of these three components to convolutional neural nets is to
build a set of feature detectors which simultaneously reduce dimensionality of
the input image, resulting in a vector containing information on the spatial
structure of the input image.

#### Autoencoders

Autoencoders are a type of artificial neural network which performs dimension
reduction in an unsupervised manner. It consists of two components, an encoder (f(x))
and a decoder (g(x')) which together perform dimension reduction (encoder), and
dimension expansion (decoder) in such a way as to minimize the error of x - x' = x - g(f(x)).

In the case of an input image, the encoder (v = F(x)) takes a matrix of pixel values
and outputs a feature vector, known as the embedded space, with highly reduced
dimensionality. The decoder (x' = G(v)) takes the feature vector as
input and outputs an image of equal dimensionality to the input image, x. The
neural net is tasked with minimizing the image reconstruction loss, which, for
a single image is x' - x = G(F(x)) - x), typically given as the mean squared error:

TODO Equation for mean squared error in Castellano and Vessio, pg 3

The result is that the autoencoder encodes
the image in the much reduced feature space, similar to a nonlinear version of Principle
Component Analysis (PCA).

<!-- The
researchers listed above take this one step further by attaching a clustering
layer to the embedded space, allowing them to jointly optimize for both reconstruction and clustering loss, which ensures that the clustering adheres to the model's pre-trained knowledge of the relevant feature spac -->

The specifics of dimension reduction component of the encoder varies based on
specific architectures. Convolutional Autoencoders, which we implement in this work,
are autoencoders that employ convolutional neural nets to define the encoding and
decoding components. The combined effect is a network which reduces the dimensionality
of the input image (autoencoder) while learning the spatially relevant components
of the input data (convolutional NN). Put another way, from an input image x, the network
learns a highly dimensionally reduced representation of x that retains spatially
relevant information, a highly desirable input source for clustering data.

### DCEC-Paint

In this work, we evaluate the clustering performance of a Deep Convolutional Embedding
Clustering (DCEC-Paint) algorithm on a set of digitized fine art. This network is
identical to the one specified by Castellano and Vessio, 2020, who made minor
adaptations to the DCEC algorithm specified by Guo et al., 2017. The network is
a convolutional autoencoder which has a clustering algorithm attached to the
latent feature space. The network is tasked with jointly optimizing for image reconstruction and
clustering loss, ensuring that clustering is performed on a reduced dimensionality, but
spatially related representation of the input image (the latent feature space). The
structure of the network is given in figure{} (reproduce from Castellano and Vessio
figure 1).

The overall motivation of DCEC-Paint is to preserve the embedded space structure
while performing clustering so as to not lose meaningful spatial structure. Guo et al.,
noted that previous deep clustering algorithms do not attempt to maintain feature
space integrity; the clustering algorithm is allowed to fully alter the feature
space, effectively throwing away previously learned meaningful features. We posit
that this is especially important for clustering digitized artworks which are
vastly more spatially complex than some example datasets, and should, in theory, store a
large amount of artistically meaningful data in the feature space.

TODO This needs to include an equation with the overall clustering loss

Below we describe the two components of the algorithm in detail.

#### Autoencoder

As shown in figure {}, the encoder expects 128 x 128 RGB image with pixel values
scaled between 0 and 1. The encoder consists of three convolutional layers which
have 32, 64, and 128 filters, respectively. In all cases the stride length is 2
and the kernel size (local receptive field) is 5x5 for the first two convolutional
layer, and 3x3 for the final layer. All layers use the ELU activation function
which was chosen by Castellano and Vessio over the proposed ReLU for quicker
learning:

Show equation of ELU if that is important.

The output of the final convolutional layer is flattened in to a vector of dimensions
32768, which is fully connected to the embedded space. The embedded space dimensions
are highly strategic as an embedded space that is too large will not sufficiently
constrain the feature reduction so no learning occurs, while a too restrictive
size will result in slow learning. We initially set this size to 32 to replicate
the value used by Castellano and Vessio, and experiment with different values.
From the embedded space, the decoder upsamples images with an architecture that
mirrors the encoder.

#### Clustering

The architecture of the clustering layer is derived from Xie et al., 2016, who
proposed the method as part of NN (Deep Embedded Clustering - DEC) which, given
a high dimensionality data space, used
Stacked Autoencoders to form a reduced dimensionality feature space, and then
optimized parameters by computing a probability distribution for membership to
a set of centroids and used stochastic gradient descent via backpropagation to
learn an mapping which minimizes the Kullback-Leibler (KL) divergence to that
distribution.

After first learning an initial feature space, we set the cluster centroids to
initial values using K-means. DEC then iteratively performs two steps:

1.  Computes the similarity between a point in the feature space, z_i, and a centroid,
    mu_j using Student's t-distribution, interpreted as the probability that
    sample i belongs to cluster j.
2.  Calculates the clustering loss via KL divergence and updates the target distribution
    for future assignments.

##### Calculation of Soft Assignment

The probability that a sample from the feature space, z_i, belongs to cluster j,
is taken to be given by Student's t-distribution, given by:

Equation (1) from Xie et al.

In this experiment, as in (cite all others), we keep alpha = 1

##### Minimizing Clustering Loss

To improve the cluster centroids, DEC attempts to match the soft assigment to
an auxillery target distribution, p_i, and measure the fit of the match via KL
Divergence:

L = KL(P||Q) = see equation 2 in Xie et al.

p_i is chosen carefully by Xie et al., to satisfy three conditions:

1. Its use will strengthen predictions
2. It assigns more emphasis to data points that have high confidence
3. It normalizes the loss contribution of each centroid so that large clusters
   do not out compete smaller clusters

p_i is calculated as:

Equation 3 in Xie et al.

L is differentiable with respect to the cluster centroids, so we can update them
using SGD. The overall effect of this clustering loss function, L, is that samples
which have a high confidence in belonging to certain cluster will contribute
largely to the gradient of L wrt that cluster centroid, resulting in a movement
in the weights toward that cluster wrt to that example.

#### Optimization

Taken together, the algorithm attempts to minimize the following objective function:

L = lambda * L_r + (1 + lambda) * L_c

Where L_r is the image reconstruction loss and L_c is the clustering loss given in
the equation above.

1.  We first perform a pre-training step to build the feature space, setting, lambda
    equal to 1.
2.  We then initialize cluster centroids using K-means
3.  Setting lambda to 0.1, we jointly optimize for clustering and reconstruction loss by:
    1.  Calculate L_c and L_r and update the autoencoder weights and cluster centers
        according to SGD via backpopagation using partial L/z_i and partial L_c/mu_i
    2.  Every T iterations, update the target distribution P, based on the prediction
        from all examples in the dataset
4.  If the percentage of points that changed cluster assigment is less than some tolerance
    in between two epochs, then the algorithm terminates.


<!-- #### Prediction, Optimization, Parameter Initialization, etc. -->
### Methods
<!-- Methods go here -->

<!-- ## Experiments -->

<!-- ### Datasets -->
<!-- Explain the following: -->
<!-- Data Acquisition -->
<!-- Data Cleaning -->
<!-- Data Preprocessing -->
<!-- Exploratory Statistics -->

### Implementation
<!-- How long was the model trained, on what architecture, how many iterations, etc -->

### Prediction Evaluation
<!-- How did we evaluate the model's performance -->
<!-- Comparison to the estimate for the painting. Those are the experts -->
<!-- Comparison to alternative methods? What would those be? Is there precedence? -->
<!-- Color Pallete, smoothness, brightness, and portrait scores -->

### Experiment Results
<!-- What did we see? -->

## Discussion
<!-- Why did we see it? -->

## Conclusion
