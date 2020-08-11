---
title: "WORKING PAPER - Deep Convolutional Embedded Clustering of Digitized Fine Art"
author:
    - Brad West
keywords: ["computer vision", "autoencoder", "clustering", "fine art"]
abstract: |
    An abstract which has yet to be written.
documentclass:
    - article
hyperrefoptions:
    - linktoc=all
    - pdfwindowui
papersize: letter
csl: /home/dubs/.csl/ieee.csl
link-citations: true
header-includes: |
    \usepackage[margin=1.25in]{geometry}
    \renewcommand{\baselinestretch}{1.5}
---

<!--
* Get BibTex working with pandoc, generating to latex
* Add entries to BibTex
* Write Draft Introduction
* Change neural network to deep neural network
* Write technical discussion of neural Networks
* Write technical discussion of Convolutional Autoencoders
* Write partial section of methods
* Select ~10k images, preferably of a manageable number of artists, of a given time, with prices
* Do some basic analysis on the artists, periods, and prices of those images
* Train DCEC-paint on those images with variable number of clusters and whatnot
* Make t-SNE diagrams
* DECISION whether you want to pursue prediction

* Cross Depiction problem:
  - How is the female form picked up? Lots of naked women in different depictions

Notes about dataset:
- More recent than other datasets because it was selected based on the top x most prolific artists
- more diverse in medium and genre than other sets
  - East asian art
  - pop art
  - modern
  - etc
- Larger number of artists
- Would be interesting to see what duer's stuff compares with, as well as the chinese painters, and lucio fontana
- Take an artists work and visualize those works relative to the other clusters

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
* Use Heading identifiers - see pandoc manual

Images
![la lune](lalune.jpg "Voyage to the moon")

footnotes
Here is a footnote reference,[^1] and another.[^longnote]
[^1]: Here is the footnote.
[^longnote]: Here's one with multiple blocks.

citations
Blah blah [@smith04; @doe99].
-->

<!-- # Deep Convolutional Autoencoder Prediction of Art Action Prices -->

<!-- ## Abstract -->

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

# Introduction
<!-- This is where your state the motivation -->
<!-- contributions to the field -->

The widespread digitization of fine art over the past two decades has coincided
with large advances in computational image analysis. As museum and gallery
collections move onto the internet, millions of artworks have been made
available for view at the click of a mouse[^google-arts-and-culture]. A
proliferation of researchers
have sought to analyze these digital collections, contributing methods for a
wide variety of computer vision tasks from classification problems (of, for
example, genre, style, artist, or historical period)[@Cetinic_et_al_2018;
@Lee_and_Cha_2016] to visual relationships
between paintings [@Garcia_et_al_2019; @Castellano_et_al_2020].

[^google-arts-and-culture]: For example, Google Arts & Culture's digitization of
hundreds of museum collections: https://artsandculture.google.com/partner.

The complete visual and emotional effect of a painting is a combination
of many factors, for example the color, texture, spatial complexity, and
contrast. An art expert
recognizes those qualities and is able to place a work in its historical
and artistic context. That task, however,
is difficult to articulate and exceedingly difficult to generalize to the set
of all artwork. Early attempts at computational art analysis, nevertheless,
attempted to build a similar model by first engineering and extracting domain specific features
from the pixel space (corners, edges, SIFT), and using those feature vectors
as input to a model[@Oliva_and_Torralba_2001]. These techniques formed the
backbone of early attempts at
object recognition within and outside the art domain, and saw some success in
evaluating art[@Shamir_et_al_2010]. In recent years, however, the field has
undergone large advancements in
computer vision techniques, in particular
Convolution Neural Nets (CNNs) which have demonstrated outstanding results in
extracting semantic meaning from digitized work[@Tan_et_al_2016]. These results are impressive
both in relation to earlier, feature engineering based attempts, and
in comparison to the perceived complexity of recognizing the distinct visual
appearance of an artwork.

Rather than using engineered features that attempt to
proxy semantic attributed, CNNs attempt to learn relevant features from a large
set of training images. CNNs applied to fine art related tasks have benefited
from both large annotated sources of art data[^wikiart], as well as enormous datasets of
non-art related images[^imagenet]. In the former case, annotated art datasets allow researchers
to train classification models without hand labeling artist and genre metadata.
This has led to a number of successful models that can identify period and even
artist [@David_et_al_2016]. In the latter case, large non-art related image datasets
have been used to pre-train object recognition models for art-related tasks.

[^wikiart]: e.g. WikiArt (https://www.wikiart.org), which contains over 130 thousand
digitized works.

[^imagenet]: e.g. ImageNet (http://image-net.org), which contains more than 14 million
hand annotated images.

The large availability of annotated datasets has led many authors to focus on
supervised learning tasks. Comparatively little art research focuses on unsupervised
learning, including clustering. Clustering artwork has a number of applications to
aid the art expert in knowledge discovery, including identifying stylistic
discontinuities within an artists career,
shared techniques between groups of artists, and distinct periods within unattributed
groups of work (e.g. ancient east Asian art). Yet clustering images has been
historically difficult due to the difficulty in defining relevant features, and
issues defining distance metrics that are effective in the high dimensionality
data space of complex images. An alternative to these two problems, is to learn
an efficient representation of the input image through a deep NN.

In this work, we replicate a CNN clustering algorithm, Deep Embedded
Convolutional Embedded Clustering (DCEC) first proposed by Guo et al., 2017[@Guo_et_al_2017]
and adapted by Castellano and Vessio, 2020[@Castellano_and_Vessio_2020] to an
art specific dataset. DCEC is composed of
two components, a convolutional autoencoder (CAE) and a clustering layer
attached to the CAE. Autoencoders perform non-linear dimension reduction of the
image in two steps: an encoder which learns a mapping from the input image (data space)
to a highly reduced latent (embedded) pixel space, and a decoder which maps
from the latent space to a reconstructed image in the data space. By attaching
a clustering layer to the CAE and simultaneously optimizing
for both clustering and image reconstruction loss, DCEC ensures that clustering
of the images occurs within the constraints of the latent space. In other words,
clusters are formed in the presence of spatial attributes deemed meaningful by their
inclusion in the latent space.

We evaluate clustering performance against two separate datasets of
digitized artwork, both scraped from Christie's' public
record of auction house sales[^christies].
<!-- The first dataset we cluster is a set of [TODO: n (b/w)] images, and the second
is a set of [TODO: n (east/asian)]. -->
The only known prior
work with this algorithm used two datasets, one of a set of paintings by 50
well known artists and the collected works of Pablo Picasso. The images used in
this work include more obscure artists, as well as a higher proportion of
intra-genre work.
<!-- which exercises the algorithm's performance in face of lower
magnitude differences in the feature space. -->

[^christies]: https://www.christies.com/Results

# Related Work
<!-- DCN efforts, specifically deep clustering, like DEC, DCEC, DCEC-Paint -->
<!-- TODO Efforts to quantify art prices, especially using extracted, not learned features -->

As institutions digitized their art collections over the preceding decades,
researchers responded by applying computational image analysis methods to the art domain.
A large number of efforts focused on classification tasks, for example of
genre[@Cetinic_et_al_2016];
object detection and recognition[@Crowley_and_Zisserman_2014];
visual similarity between artwork[@Castellano_et_al_2020; @Seguin_et_al_2016],
and the cross depiction problem -- distinguishing the same type of object in different
representations, say, a cat depicted in a cubist vs an impressionist painting[@Hall_et_al_2015].

Initial attempts to analyze art emphasized feature engineering and extraction
in which domain specific characteristics of artwork are
identified and a given artwork's relative presence or absence of those
features is used as input to a model. For example, Oliva and Torralba[@Oliva_and_Torralba_2001]
proposed a set of "perceptual dimensions (naturalness, opennes, roughness,
expansion, ruggedness)" which they estimated using low level pixel relationships
and used to categorize scenes. Shamir et al.[@Shamir_et_al_2010] used 11 extracted
features to classify paintings by their artists and art school. Spehr et al.[@Spher_et_al_2009]
used over 200 features to cluster 10,000 paintings.

While feature engineering has been shown to be effective, it's limited by its
requirement for a comprehensive set of pre-identified features for modeling the
image characteristics. This shortcoming is especially apparent in tasks which
attempt to consider the image as a whole, a task which seems especially complex
given the myriad possible "features" that could be found in an m x n x p
dimensioned image. Over the past decade, the computer
vision community has focused on designing algorithms which, rather than rely on
extracted features, learn a relevant feature set through a training process.
Applying deep learning concepts to art analysis has proved fruitful
in a host of subfields, often employing convolutional neural nets
[@Cetinic_et_al_2016; @Cetinic_et_al_2018;
Crowley_and_Zisserman_2014; @Tan_et_al_2016; @Garcia_et_al_2019].

Due to the availability of numerous and large labeled dataset, much research uses
supervised learning methods.
In recent years,
however, a few authors have focused on unsupervised learning, in particular
clustering. Seguin et al.[@Seguin_et_al_2016] used a convolutional neural net to
cluster images for visual link retrieval. Outside the art domain,
Xie et al.,[@Xie_et_al_2017] proposed a Deep Embedded
Clustering (DEC) algorithm which selects clusters in two steps: first learning a reduced
dimensionality set of features using stacked autoencoders (SAE) and second
using stochastic gradient descent (SGD) to learn cluster centers
Guo et. al[@Guo_et_al_2017] expanded on this work by using Convolutional Autoencoder
rather than a stacked auto encoder, and by jointly optimizing for both clustering
and image reconstruction loss so as to avoid corrupting the reduced
dimensionality feature space during the clustering component, naming the
algorithm Deep Convolutional Embedded Clustering (DCEC).
Inspired by DCEC, Castellano and Vessio[@Castellano_and_Vessio_2020]
adapted it to more complex and larger images in the art domain and demonstrated its
efficacy in clustering a dataset of ca. 10,000 digitized artworks.

# DCEC Architecture
<!-- Motivation for what NNs offer in general -->
<!-- What do NN offer to image problems -->
<!-- What do they offer to this specific problem -->

<!-- 
* What is an image?
    - An image is a set of pixels, each with three dimensions, arrayed in a grid.
    - The given arrangement of these pixels is what gives these images their texture/shape/complexity, etc
 -->

Before explaining the architecture of the DCEC-Paint algorithm, we provide a
brief and incomprehensive overview of the mathematical concepts underlying
CNNs and Autoencoders. For a textbook introduction to artificial neural nets,
see [@Engelbrecht_2007].

<!-- Over the past decade, researchers have used convolutional neural networks to
investigavision problems such as object
detection and facial recognition [cite here]. Autoencoders are a type of unsupervised neural
net which learns a mapping from a high dimensional data space to a lower dimensional
feature space. Like CNNs, Autoencoders have proved wildly useful in the computer vision
domain due the high dimensionality and complexity of images [cite here]. This
section provides a brief and incomprehensive overview of the mathematical
concepts underlying CNNs and Autoencoders. For a gentle introduction to neural
networks and deep learning, see [Michael Nielson]. For a comprehensive look at
neural networks including autoencoders, see [cite textbook]. -->

## Artificial Neural Networks
<!-- How and why do NNs work -->

### Architecture

<!-- TODO This paragraph does not read very well -->
Artificial neural networks are non-linear functions,
$F_{nn}: \mathbb{R}^I -> \mathbb{R}^K$, where $I$ and
$K$ are the dimensionality of the input and output spaces, respectively.
Modeled after their biological equivalents, they achieve this non-linear functionality
through composition of layers of artificial neurons where an individual neuron
is itself typically a nonlinear function,
$y = f(x + b)$, (almost always $y = [0, 1]$ or $y = [-1, 1]$).
$y$ is called an activation function, accepting n > 0 input signals (x) and
outputing a single value as a function of the inputs and the learned weights
(w) and biases (b) for each inter-neuron connection.

**TODO** - Figure of an artificial neuron with input and output signals, annotated
to match equations above.
<!-- [diagram here, similar to figure 2.1 in englebrecht]. -->

Thus, for each edge (connection) between the $j^{th}$ neuron in layer $i$ and
the $k^{th}$ neuron in
layer $i+1$, the network learns a particular weight ($w_{i+1,j,k}$) which represents
the relative importance of that component of the total input signal. When
layered together, the output of a neuron in layer $i$ is the input to
neurons in layer $i + 1$, forming a structure similar to the one depicted in the
figure x[^nn].
<!-- Above, do:
![This is the caption\label{mylabel}](/url/of/image.png)
See figure \ref{mylabel}. -->

**TODO** - Figure of a generic feedforward network

[^nn]: While typically a directed acyclic graph (DAG), the actual structure of
a particular type of DNN is highly variable
within these general constraints. The activation function, number of inputs, number
of layers, interconnectedness of the layers, and even direction of network connections
can all vary to form different networks.

<!-- Activation functions -->
In general, an activation function is a monotonically increasing function,
$F_{AN}: \mathbb{R} -> [0, 1]$ or $F_{AN}: \mathbb{R} -> [-1, 1]$
such that:

$$
F_{AN}(-\infty) = 0 \quad \textrm{or} \quad F_{AN}(-\infty) = 1
$$

and

$$
f_{AN}(\infty) = 1
$$

There are many viable activation functions, but we will focus on the rectifier,
specifically the exponential linear unit (ELU). The ELU and variants such as
the rectified linear unit (ReLU)
have been shown to speed up training on large and deep neural networks compared
to more traditional choices such as the logistic sigmoid and hyperbolic tangent
(need citation here). ELU:

$$
  f(x)=\left\{
  \begin{array}{@{}ll@{}}
    x, & \text{if}\ x>0 \\
    a(e^x - 1), & \text{otherwise}
  \end{array}\right.
$$

where $a$ is a constant which can be tuned.

This function has highly beneficial properties for learning. It is continuously
differentiable along the real numbers, which, combined with the fact that it is
monotonically increasing means that a gradient can be calculated at any place,
facilitating "learning" as discussed in the section below. The specifics of the
ELU compared to its variants such as the ReLU, and leaky ReLU, are too specific
to address here. It has been shown, however, to speed up learning on deep networks
as well as lead to better classification accuracy[@Clevert_2015].

### Artificial Learning

Neural networks with at least a single hidden layer can be shown to approximate
any continuous function[@Hornik_1991]. NNs achieve this impressive result by "learning"
the appropriate weights and biases, progressively updating these values until
an approximation is sufficiently close. There are many different learning algorithms,
the components for updating the weights ($w$) and biases ($b$) on the networks
signals. In the interest of brevity, we
focus on the most widely used learning rule, Stochastic Gradient Descent (SGD)
with backpropagation. For an introduction
to additional rules such as the Widrow-Hoff, Generalized Delta, and Error-Correction
learning rules, see [@Engelbrecht_2007].

#### Stochastic Gradient Descent

Consider a fully connected 3 layer feed forward neural net,
depicted in figure (above). Each neuron in the ith layer is connected to each
neuron in layer i + 1 and inputs are passed forward (that is, right to left)
through the network.

Stochastic Gradient Descent (SGD) attempts to minimize the value of an error
function (also known as an optimization or cost function], $\mathcal{E}(y - y')$,
by progressively updating the network weights ($w$) and biases ($b$) in such a
way as to follow the first derivative gradient of $\mathcal{E}$ with respect to
the weights and biases ($\frac{\partial{\mathcal{E}}}{\partial{w}}$,
$\frac{\partial{\mathcal{E}}}{\partial{b}}$). By
iteratively calculating these gradients and then updating the weights and biases
in that direction, the network progressively learns the appropriate weights for
function approximation.

<!-- For example, consider a common form for epsilon, the sum of squared errors:

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
that results in the largest reduction in Epsilon. -->

#### Backpropagation

Despite knowing the gradient, however, it remains unclear how each weight in the network
is updated. This is achieved via backpropagation, first appreciated by [@Rumelhart_et_al_1986].
The details are too specific to include here, but the algorithm has two general
steps:

1.  Inputs are passed through the network and output values calculated, providing
    the values needed to calculate epsilon and the derived gradients.
2.  The error signal from the output layer is propagated backwards through the
    network such that each weight and bias is updated according to its effect on
    the overall error of the cost function.

(**TODO** - need to explain back propagation in more detail here)

## Convolutional Autoencoders

**TODO** - Some relevant art image

<!-- TODO this should have a particular image associated with it, something famous -->
While to the human eye the image above may appear a mosaic of colorful brush
strokes, masterfully arranged to elicit emotion,
digitally, images are represented as matrices of numeric values pertaining to pixel
intensity. In the case of black and white images, there is a single channel,
meaning an image can be represented by a single matrix of dimensions $m \times n$.
Color images are $3$ dimensional, accounting for the increase from one to p > 1
channels (3, in the case of RGB images).

Consider a neural network tasked with learning relevant features
from an input dataset of $k$, $m \times n \times p$ images. Intuitively, a network might seem
like an appropriate tool for identifying
relationships between pixels. An architecture of many interconnected neurons
seems as though it should have the ability to progressively "learn" more meaningful
patterns in the data (over the course of many epochs) by progressively weighting
certain neuronal relationships. Convolutional neural nets build on this intuition
by building multiple layers that attempt to detect features at different abstraction
layers, and which contain multiple filters in each layer, each filter targeting
a different "feature".

### Convolutional Neural Networks

Convolutional neural networks are a type of deep neural network[^DNN]
with an architecture that is well suited
to image analysis. In particular, convolutional neural nets are better able to
extract the spatial structure of an image than traditional networks because they
(1) limit the connectectedness of the network by ensuring connected neurons correspond
to spatially adjacent input pixels, (2) share weights among edges within the same
layer, and (3) pool subsequent layers to provide dimension reduction. For
a full discussion of convolutional neural nets, see
[@LeCun_et_al_1998]. We summarize briefly here.

[^DNN]: Deep neural networks contain more than 2 hidden layers.

Consider the $4 \times 4$ square input image depicted below.

**TODO** - image of 4 x 4 x 1 pixels

In a fully connected network, each pixel value would be input to each neuron in the
first hidden layer. Convolutional neural networks differ from fully connected
networks by defining a local receptive
field of size $k$, where ($k \times k$) is the number of adjacent inputs that will connect to
the $j^{th}$ neuron in layer $i$.

**TODO** - figure of the 4x4 input mapping with a 2x2 receptive field to a 3x3 input

For each neuron in the hidden layer, this local receptive field is moved adjacently
by a stride length of $l$ pixels. For example, a stride length of 2 will correspond to a total
of 9 local receptive fields, meaning a total of 9 neurons in the first hidden layer.
By limiting the number of input signals passed to a single neuron, each neuron
receives information only from adjacent pixels, rather than information from
every input in the network, linking pixels/neurons that are spatially distant.
Intuitively, this seems like an appropriate way to extract spatial meaning.

Furthermore, by design, each neuron in the $i^{th}$ hidden layer has the same weights
and biases. So, for our example, the $2 \times 2$ array of weights and biases input to the
jth neuron in the ith layer are identical to every other array of weights and biases
in the ith layer. By making this restriction, the network ensures that all the neurons
in the ith layer are detecting the same spatial structure. This map from an input
layer to a hidden layer is called a feature map, and the set of weights and biases
that define a feature map is called a filter or a kernel (**TODO** - cite something here).
By increasing the number of feature maps at each layer, the network is able to
detect multiple features[@Bengio_et_al_2013].

Finally, convolutional NNs aggregate the activations of the convolution layers
in what are known as pooling layers. By pooling adjacent activations, typically by taking
the maximum activation from a $l \times l$ sized area, the feature maps undergo
dimension reduction. Intuitively, this can be thought of as a function which outputs
whether a certain feature is found anywhere within a subsection of a layer.

The combined effect of these three components to convolutional neural nets is to
build a set of feature detectors which simultaneously reduce dimensionality of
the input image, resulting in a vector containing information on the spatial
structure of the input image.

### Autoencoders

Autoencoders are a type of unsupervised artificial neural network consisting of
two components, an encoder, $x' = f(x)$,
and a decoder $g(x')$, which perform, respectively, dimension reduction and
expansion in such a way as to minimize the error of
$y - y' = y - g(f(x))$, where $x'$ is a projection of $x$ from the data space into
a much lower dimensional latent space. Figure \_ shows a rudimentary autoencoder.

**TODO** - Figure of a basic feedforward autoencoder

Autoencoders have proved especially useful in denoising and dimension reduction
of images, which are naturally highly dimensional. Typically the reconstruction
loss function is taken to be the mean squared error:

$$
L = \frac{1}{n}\sum_{i=1}^{n} (x'_i - x_i)^2 = \frac{1}{n}\sum_{i=1}^{n} (g(f(x)) - x_i)^2
$$

The result is that the autoencoder encodes the image in the much reduced
feature space, similar to a nonlinear version of Principle Component Analysis
(PCA).

The specifics of the dimension reduction component of the encoder varies based on
specific architectures. Convolutional Autoencoders, which we implement in this work,
are autoencoders that choose convolutional neural nets as encoders and decoders.
The combined effect is a network which reduces the dimensionality
of the input image (autoencoder) while learning the spatially relevant components
of the input data (convolutional NN). Put another way, from an input image x, the network
learns a highly dimensionally reduced representation of x that retains spatially
relevant information, a highly desirable input source for clustering data.

## DCEC-Paint

In this work, we evaluate the clustering performance of the Deep Convolutional
Embedded Clustering (DCEC-Paint) algorithm on a set of digitized fine art. This network is
identical to the one specified by [@Castellano_and_Vessio_2020], who made minor
adaptations to the DCEC algorithm specified by [@Guo_et_al_2017]. A
convolutional autoencoder with a deep learning clustering algorithm feed from the
latent feature space, the network is tasked with jointly optimizing for image reconstruction and
clustering loss, ensuring that clustering is performed on a reduced dimensionality, but
spatially related representation of the input image. The
structure of the network is shown in figure \_.

**TODO** - Figure showing DCEC-Paint, similar to figure 1 in Castellano and Vessio

The overall motivation of DCEC-Paint is to preserve the embedded space structure
while performing clustering so as to not lose meaningful spatial structure. [@Guo_et_al_2017]
noted that previous deep clustering algorithms do not attempt to maintain feature
space integrity; the clustering algorithm is allowed to fully alter the feature
space, effectively throwing away previously learned meaningful features. We posit
that retaining learned features is especially important for clustering digitized
artworks which are, on the whole, spatially complex, and should in theory store a
large amount of artistically meaningful data in the feature space.

As mentioned above, the network is tasked with minimizing the overall loss, a
combination of the image reconstruction and clustering loss:

$$
L = \gamma L_r + (1 - \gamma) L_c
$$

where $\gamma$ is a tunable parameter. [@Guo_et_al_2017] set $\gamma = 0.1$,
while [@Castellano_and_Vessio_2020] used $\gamma = 0.9$, putting more importance
on optimizing for clustering loss rather than reconstruction loss. The goal of
clustering artwork is to optimize cluster assignment, so we follow [@Castellano_and_Vessio_2020]
and set $\gamma = 0.9$

Below we describe the derivation of the reconstruction and clustering loss in
detail, with their component parts.

### Autoencoder

As shown in figure \_, the encoder expects $128 \times 128$ RGB image with pixel values
scaled between 0 and 1. The encoder consists of three convolutional layers which
have 32, 64, and 128 filters, respectively. In all cases the stride length is 2 pixels
and the kernel size (local receptive field) is $5 \times 5$ for the first two convolutional
layers, and $3 \times 3$ for the final layer. All layers use the ELU activation function.

The output of the final convolutional layer is flattened in to a vector of size
$327684, which is fully connected to the embedded space. Choosing the embedded space dimensions
is highly strategic as an embedded space that is too large will not sufficiently
constrain the feature reduction, resulting in minimal learning, while too restrictive a
size will result in slow learning. We initially set this size to 32 to replicate
the value used by Castellano and Vessio, and experiment with different values.
From the embedded space, the decoder upsamples images with an architecture that
mirrors the encoder.

### Clustering

The architecture of the clustering layer is derived from [@Xie_et_al_2017], who
introduced the method as part of Deep Embedded Clustering (DEC) which, given
a high dimensionality data space, used
stacked autoencoders to form a reduced dimensionality feature space, and then
optimized parameters by computing a probability distribution for membership to
a set of centroids and used stochastic gradient descent via backpropagation to
learn an mapping which minimizes the Kullback-Leibler (KL) divergence to that
distribution.

After first learning an initial feature space, setting $\gamma = 0$ in the
overall loss function, we set the cluster centroids to
initial values using K-means and set $\gamma = 0.9$. DEC then iteratively
performs two steps learn cluster weights:

1.  Computes the similarity between a point in the feature space, $z_i$, and a centroid,
    $\mu_j$ using Student's t-distribution, interpreted as the probability that
    sample $i$ belongs to cluster $j$.
2.  Calculates the clustering loss via KL divergence and updates the target distribution
    for future assignments.

#### Calculation of Soft Assignment

The probability that a sample from the feature space, $z_i$, belongs to cluster $j$,
is taken to be given by Student's t-distribution with one degree of freedom,
given by:

$$
q_{ij} = \frac{(1 + (z_i - \mu_j)^2)^{-1}}{\sum_{j'}(1 + (z_i - \mu_{j'})^2)^{-1}}
$$

[@Xie_et_al_2017] use a t-distribution after [@Maaten_et_al_2008], who introduced
it for t-SNE, a technique for visualizing high dimensional spaces in two or three
dimensions.

**TODO** - Details

#### Minimizing Clustering Loss

To improve the cluster centroids, DEC attempts to match the soft assignment to
an auxillary target distribution, $p_i$, and measure the fit of the match via KL
Divergence:

$$
L = KL(P||Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$
<!-- TODO you need to build that out -->

$p_i$ is chosen carefully in [@Xie_et_al_2017], to satisfy three conditions:

1. Its use will strengthen predictions
2. It assigns more emphasis to data points that have high confidence
3. It normalizes the loss contribution of each centroid so that large clusters
   do not out compete smaller clusters

$p_i$ is calculated as:

$$
p_{ij} = \frac{q_{ij}^2/f_j}{\sum_{j'} g_{ij'}^2/f_{j'}}
$$

L is differentiable with respect to the cluster centroids, so we can update them
using SGD. The overall effect of this clustering loss function, L, is that samples
which have a high confidence in belonging to certain cluster will contribute
largely to the gradient of L with respect to that cluster centroid, resulting in a movement
in the weights toward that cluster for that example

#### Optimization

The full DCEC-Paint algorithm proceeds as follows:

1.  We first perform a pre-training step to build the feature space, setting, $\gamma$
    equal to 1.
2.  We then initialize cluster centroids using K-means
3.  Setting $\gamma$ to 0.1, we jointly optimize for clustering and reconstruction loss by:
    1.  Calculating $L_c$ and $L_r$ and updating the autoencoder weights and cluster centers
        according to SGD via backpopagation using $\frac{\partial L}{\partial z_i}$ and $\frac{\partial L_c}{\partial \mu_i}$
    2.  Every $T$ iterations, update the target distribution $p$, based on the prediction
        from all examples in the dataset
4.  If the percentage of points that changed cluster assignment is less than some tolerance
    in between two epochs, then the algorithm terminates.

<!-- #### Prediction, Optimization, Parameter Initialization, etc. -->
# Methods

## Dataset

Christie's publicizes online the results of all public auctions held
after December, 2005. This amounts to a truly impressive record of fine art,
furniture, antiques, jewelry, and other collectible sales; well over half a
million pieces auctioned over hundreds of sales held across the world and
online. As one of world's two leading auction houses (the other being Sotheby's),
these auction results contain reams of influential and expensive art, including
the most expensive piece ever sold, da Vinci's Salvator Mundi[^mundi], which
sold for an astounding $450 million.

[^mundi]: https://www.christies.com/lotfinder/paintings/leonardo-da-vinci-salvator-mundi-6110563-details.aspx

To obtain our final dataset we scraped the Christie's website for all auctions
that consisted primarily of artwork in two dimensional mediums. Left with a
set of nearly 300k works of art, we chose the top 50 most prolific artists and,
randomly selecting 250 works of art for each artist, we hand curated the dataset
to exclude intermediate sketches and sculptures. The final dataset contains
$n = 10,505$ 3 channel $128 \times 128$ images which span a diverse set of
artists, mediums, and movements (see figure).

Compared to previous applications of this algorithm, we believe this dataset
to represent a much broader set of artists and mediums. For instance, there are
number of photographs included in the dataset, including varied works by
Ansel Adams, Henri Cartier-Bresson, and Hiroshi Sugimoto. Within paintings,
there are the late 19th century Chinese masters, Pu Ru and Zhang Daqian, along
with Pablo Picasso, Salvador Dali, and Rembrant. Thus, the algorithm is tasked
with clustering works across artistic style and medium.

With an artistically diverse dataset, we expect the algorithm to be particularly
exposed to the cross depiction problem. Imagine two digitized works depicting
a clock but of different artistic styles and mediums. One, Salvador Dali's famous
1931 "Persistence of Memory", and the other a moment captured by the
lens of Henri Cartier-Bresson. Holding all else equal, we expect a "correct"
clustering algorithm to find these two works more similar than a separate,
hypothetical, pair of works that don't contain clocks. The algorithm's ability
on these qualitative tasks is additionally important to the quantitative measures
outlined below.

**TODO** - This would be a good location for some of the images

<!-- TODO - You could include a plot with distribution of image prices -->

| artist                       | n_images | birth | death | mediums                     | movement                             |
|------------------------------|----------|-------|-------|-----------------------------|--------------------------------------|
| henri cartier-bresson        | 250      | 1908  | 2004  | photography                 |                                      |
| victor vasarely              | 250      | 1906  | 1997  | painting,sculpture          |                                      |
| ansel adams                  | 249      | 1902  | 1984  | photography                 |                                      |
| sam francis                  | 249      | 1923  | 1994  | painting,printmaking        |                                      |
| hiroshi sugimoto             | 249      | 1948  |       | photography                 |                                      |
| maurice de vlaminck          | 246      | 1876  | 1958  | painting                    |                                      |
| wayne thiebaud               | 246      | 1920  |       | painting                    |                                      |
| karel appel                  | 244      | 1921  | 2006  | painting                    |                                      |
| andy warhol                  | 243      | 1928  | 1987  | painting                    | pop art                              |
| bernard buffet               | 242      | 1928  | 1999  | panting,drawing,printmaking |                                      |
| keith haring                 | 242      | 1958  | 1990  | pop art,street art          |                                      |
| gerhard richter              | 241      | 1932  |       | photography,painting        |                                      |
| jasper johns                 | 239      | 1930  |       | painting,pop art            |                                      |
| alighiero boetti             | 238      | 1940  | 1994  | painting                    |                                      |
| robert motherwell            | 238      | 1915  | 1991  | painting,printmaking        |                                      |
| marc chagall                 | 237      | 1887  | 1985  | cubism,expressionism        |                                      |
| helmut newton                | 236      | 1920  | 2004  | photography                 |                                      |
| jean dubuffet                | 236      | 1901  | 1985  | painting                    |                                      |
| irving penn                  | 235      | 1917  | 2009  | photographer                |                                      |
| robert rauschenberg          | 234      | 1925  | 2008  | painting                    |                                      |
| jim dine                     | 233      | 1935  |       | painting                    |                                      |
| joan miro                    | 233      | 1893  |       | painting                    |                                      |
| frank stella                 | 233      | 1936  |       | painting,printmaking        |                                      |
| christo                      | 232      | 1935  |       | painting                    |                                      |
| tom wesselmann               | 232      | 1931  | 1945  | painting                    |                                      |
| takashi murakami             | 230      | 1962  |       | contemporary art            |                                      |
| roy lichtenstein             | 229      | 1923  | 1997  | painting                    |                                      |
| sol lewitt                   | 229      | 1928  | 2007  | painting,drawing            |                                      |
| zao wou-ki                   | 228      | 1920  | 2013  | painting                    |                                      |
| damien hirst                 | 227      | 1965  |       | painting                    |                                      |
| raoul dufy                   | 226      | 1877  | 1953  | painting                    |                                      |
| qi baishi                    | 222      | 1864  | 1957  | painting                    |                                      |
| david hockney                | 213      | 1937  |       | pop art                     |                                      |
| zhang daqian                 | 210      | 1899  | 1983  | painting                    |                                      |
| laurence stephen lowry, r.a. | 209      | 1887  | 1976  | painting                    |                                      |
| pierre-auguste renoir        | 203      | 1841  | 1919  | painting                    |                                      |
| alexander calder             | 202      | 1898  | 1976  | sculpture                   |                                      |
| francis newton souza         | 199      | 1924  | 2002  | painting,drawing            |                                      |
| max ernst                    | 198      | 1891  | 1976  | painting                    | dada,surealism                       |
| albrecht dürer               | 191      | 1471  | 1528  | painting,printmaking        |                                      |
| pu ru                        | 179      | 1896  | 1963  | painting                    |                                      |
| lucio fontana                | 174      | 1899  | 1968  | painting                    |                                      |
| salvador dalí                | 173      | 1904  | 1989  | painting                    | cubism,dada,surrealism               |
| rembrandt harmensz. van rijn | 171      | 1606  | 1669  | painting                    | dutch golden age,baroque             |
| henry moore                  | 148      | 1898  | 1986  | bronze scuplture            | modernsim                            |
| henri de toulouse-lautrec    | 120      | 1864  | 1901  | painting                    | post-impressionism,Art Noveau        |
| pablo picasso                | 117      | 1881  | 1973  | painting                    | cubism,surrealism                    |
| henri matisse                | 99       | 1869  | 1954  | painting                    | fauvism,modernism,post-impressionism |
| edgar degas                  | 83       | 1834  | 1917  | painting                    | impressionism                        |
| auguste rodin                | 18       | 1840  | 1917  | drawing                     |                                      |
<!-- Methods go here -->

<!-- ## Experiments -->

<!-- ### Datasets -->
<!-- Explain the following: -->
<!-- Data Acquisition -->
<!-- Data Cleaning -->
<!-- Data Preprocessing -->
<!-- Exploratory Statistics -->

# Implementation
<!-- How long was the model trained, on what architecture, how many iterations, etc -->

<!-- # Prediction Evaluation -->
<!-- How did we evaluate the model's performance -->
<!-- Comparison to the estimate for the painting. Those are the experts -->
<!-- Comparison to alternative methods? What would those be? Is there precedence? -->
<!-- Color Pallete, smoothness, brightness, and portrait scores -->

# Clustering Evaluation
<!-- How did we evaluate the performance of the network? -->

# Experiment Results
<!-- What did we see? -->

# Discussion
<!-- Why did we see it? -->

# Conclusion


# References
