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
    \usepackage{pdflscape}
    \usepackage{setspace}
    \doublespacing
    \usepackage[normalem]{ulem}
    \useunder{\uline}{\ul}{}
---

<!-- \renewcommand{\baselinestretch}{1.5} -->
<!--
* Change neural network to deep neural network
* Write technical discussion of neural Networks
* Write technical discussion of Convolutional Autoencoders
* Do some basic analysis on the artists, periods, and prices of those images
* Train DCEC-paint on those images with variable number of clusters and whatnot
    - How do you hypertune parameters
    - What do the results look like? Do they look good? How do they look compared to Mnist
    - How do you stop training when they are OK?
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
example, genre, style, artist, or historical period) [@Cetinic_et_al_2018;
@Lee_and_Cha_2016] to visual relationships
between paintings [@Garcia_et_al_2019; @Castellano_et_al_2020].

[^google-arts-and-culture]: For example, Google Arts & Culture's digitization of
hundreds of museum collections: https://artsandculture.google.com/partner.

![*Femme au chapeau* (Woman with a Hat), Henri Matisse, 1905, Oil on canvas [@matisse_1905].\label{woman_with_a_hat}](img/woman_with_a_hat.jpg){ width=50% }

The complete visual and emotional effect of a painting is a combination
of many factors, for example the color, texture, spatial complexity, and
contrast.
While to the human eye a work like Figure \ref{woman_with_a_hat} may appear a mosaic of colorful brush
strokes, masterfully arranged to elicit emotion,
digitally, images are represented as matrices of numeric values pertaining to pixel
intensity. In the case of black and white images, there is a single channel,
meaning an image can be represented by a single matrix of dimensions $m \times n$.
Color images are three dimensional, accounting for the increase from one to $p > 1$
channels ($3$, in the case of RGB images), resulting in an $m \times n \times p$
datum for each image.
An art expert
recognizes the artistic qualities of an image and is able to place a work in
its historical
and artistic context.
That task, however,
is difficult to articulate and exceedingly difficult to generalize to the set
of all artwork.
Early attempts at computational art analysis, nevertheless,
attempted to build a similar model by first engineering and extracting domain specific features
from the pixel space (corners, edges, SIFT), and using those feature vectors
as input to a model [@Oliva_and_Torralba_2001]. These techniques formed the
backbone of early attempts at
object recognition within and outside the art domain, and saw some success in
evaluating art [@Shamir_et_al_2010]. In recent years, however, the field has
undergone large advancements in
computer vision techniques, in particular
Convolution Neural Nets (CNNs) which have demonstrated outstanding results in
extracting semantic meaning from digitized work [@Tan_et_al_2016]. These
results are impressive both in relation to earlier, feature engineering based attempts, and
in comparison to the perceived complexity of recognizing the distinct visual
appearance of an artwork.

Rather than using engineered features that attempt to
proxy semantic attributes, CNNs attempt to learn relevant features from a large
set of training images. CNNs applied to fine art related tasks have benefited
from both large annotated sources of art data[^wikiart], as well as enormous
datasets of non-art related images[^imagenet]. In the former case, annotated art
datasets allow researchers to train classification models without hand
labeling artist and genre metadata.
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
data space of complex images. An alternative to these two problems is to learn
an representation of the input image through a deep NN.

In this work, we replicate a CNN clustering algorithm, Deep Embedded
Convolutional Embedded Clustering (DCEC) first proposed by Guo et al.
[@Guo_et_al_2017]
and adapted by Castellano and Vessio [@Castellano_and_Vessio_2020] to an
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

We evaluate clustering performance against a dataset of
digitized artwork, scraped from the public
record of auction house sales at Christie's[^christies].
The only known prior
work with this algorithm used two datasets: (1) a set of paintings by 50
well known artists and (2) the collected works of Pablo Picasso. The data used
in this work include a broader range of mediums and genres, including a large
set of photographs in addition to paintings, drawings, and prints. This additional
diversity allows us to assess the algorithm's performance across more artistic
styles and scene depictions.
<!-- which exercises the algorithm's performance in face of lower
magnitude differences in the feature space. -->

[^christies]: https://www.christies.com/Results

# Related Work
<!-- DCN efforts, specifically deep clustering, like DEC, DCEC, DCEC-Paint -->
<!-- TODO Efforts to quantify art prices, especially using extracted, not learned features -->

As institutions digitized their art collections over the preceding decades,
researchers responded by applying computational image analysis methods to the art domain.
A large number of efforts focused on classification tasks, for example of
genre [@Cetinic_et_al_2016];
object detection and recognition [@Crowley_and_Zisserman_2014];
visual similarity between artwork [@Castellano_et_al_2020; @Seguin_et_al_2016],
and the cross depiction problem -- distinguishing the same type of object in different
representations, say, a cat depicted in a cubist vs an impressionist
painting [@Hall_et_al_2015].

Initial attempts to analyze art emphasized feature engineering and extraction
in which domain specific characteristics of artwork are
identified and a given artwork's relative presence or absence of those
features is used as input to a model. For example, [@Oliva_and_Torralba_2001]
proposed a set of "perceptual dimensions (naturalness, openness, roughness,
expansion, ruggedness)" which they estimated using low level pixel relationships
and used to categorize scenes. [@Shamir_et_al_2010] used 11 extracted
features to classify paintings by their artists and art school. [@Spher_et_al_2009]
used over 200 features to cluster 10,000 paintings.

While feature engineering has been shown to be effective, it's limited by its
requirement for a comprehensive set of pre-identified features for modeling the
image characteristics. This shortcoming is especially apparent in tasks which
attempt to consider the image as a whole, a task which seems especially complex
given the myriad possible "features" that could be found in an
$m \times n \times p$
dimensioned image, where $m$, $n$, and $p$ are the pixel height, pixel width,
and number of color channels of the image, respectively. Over the past
decade, the computer
vision community has focused on designing algorithms which, rather than rely on
extracted features, use neural networks to learn a relevant feature set.
Applying deep learning concepts to art analysis has proved fruitful
in a host of subfields, often employing convolutional neural nets
[@Cetinic_et_al_2016; @Cetinic_et_al_2018;
Crowley_and_Zisserman_2014; @Tan_et_al_2016; @Garcia_et_al_2019].

Due to the availability of large labeled datasets, much research has employed
supervised learning methods. In recent years,
however, a number of studies have focused on unsupervised learning, in particular
clustering. Seguin et al.[@Seguin_et_al_2016] used a convolutional neural net to
cluster images for visual link retrieval. Outside the art domain, a number of
neural network architectures have been developed for clustering images.
[@Min_et_al_2018] surveys these methods, comparing the relative advantages of
each algorithm class. Relevant to this work,
Xie et al. [@Xie_et_al_2017] proposed a Deep Embedded
Clustering (DEC) algorithm which selects clusters in two steps by first
learning a reduced
dimensionality set of features using stacked autoencoders (SAE) and then
using stochastic gradient descent (SGD) to learn cluster centers.
Guo et. al [@Guo_et_al_2017] expanded on this work by using a convolutional
autoencoder
rather than a stacked autoencoder, and by jointly optimizing for both clustering
and image reconstruction loss so as to avoid corrupting the reduced
dimensionality feature space during clustering, naming the
algorithm Deep Convolutional Embedded Clustering (DCEC).
Inspired by DCEC, Castellano and Vessio [@Castellano_and_Vessio_2020]
adapted it to more complex and larger images in the art domain and demonstrated its
efficacy in clustering a dataset of approximately 10,000 digitized artworks.

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
brief (and not at all comprehensive) overview of the mathematical concepts underlying
CNNs and Autoencoders. For an introduction to artificial neural networks,
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
$F_{nn}: \mathbb{R}^I \rightarrow \mathbb{R}^K$, where $I$ and
$K$ are the dimensionality of the input and output spaces, respectively.
Modeled after their biological equivalents, they achieve this non-linear functionality
through composition of layers of artificial neurons where an individual neuron
is itself typically a nonlinear function,
$y = f(x + b)$, (almost always $y = [0, 1]$ or $y = [-1, 1]$).
$y$ is called an activation function, accepting $n > 0$ input signals ($x$) and
outputting a single value as a function of the inputs and the learned weights
($w$) and biases ($b$) for each inter-neuron connection.

![An artificial neuron with input signals ($z_i$) which are transformed by networks weights ($w_i$) and in put to the neuron activation function ($\sigma$). In a layered network the output of the nueron is passed to the neurons in the next layer.\label{artificial_neuron}](img/artificial_neuron.png){ width=80% }

<!-- [similar to figure 2.1 in englebrecht]. -->

Figure \ref{artificial_neuron} shows an artificial neuron with input and output
signals. For each edge (connection) between the $j^{th}$ neuron in layer $i$ and
the $k^{th}$ neuron in
layer $i+1$, the network learns a particular weight ($w_{i+1,j,k}$) which represents
the relative importance of that component of the total input signal. When
layered together, the output of a neuron in layer $i$ is the input to
neurons in layer $i + 1$, forming a structure similar to the one depicted in
Figure \ref{feedforward_net}[^nn].
<!-- Above, do:
![This is the caption\label{mylabel}](/url/of/image.png)
See figure \ref{mylabel}. -->

![A simple feedforward neural network attained by chaining layers of artificial neurons. In this network, every neuron in layer $i$ is connected to every neuron in layer $i + 1$.\label{feedforward_net}](img/ffn.png){ width=100% }

[^nn]: While typically a directed acyclic graph (DAG), the architectures of DNNs
are highly variable
within these general constraints. The activation function, number of inputs, number
of layers, interconnectedness of the layers, and even direction of network connections
can all vary.

<!-- Activation functions -->
In general, an activation function is a monotonically increasing function,
$F_{AN}: \mathbb{R} \rightarrow [0, 1]$ or $F_{AN}: \mathbb{R} \rightarrow [-1, 1]$
such that:

\begin{equation} \label{eq:act_left}
F_{AN}(-\infty) = 0 \quad \textrm{or} \quad F_{AN}(-\infty) = 1
\end{equation}
and
\begin{equation} \label{eq:act_right}
f_{AN}(\infty) = 1.
\end{equation}

There are many viable activation functions including traditional choices such as
the logistic and hyperbolic tangent functions. In this work, we use the exponential linear
unit (ELU). The ELU is defined as

\begin{equation} \label{eq:elu}
  f(x)=\left\{
  \begin{array}{@{}ll@{}}
    x, & \text{if}\ x>0 \\
    a(e^x - 1), & \text{otherwise},
  \end{array}\right.
\end{equation}

where $a$ is a constant which can be tuned.

This function has highly beneficial properties for learning. Like other commonly
used activation functions, it is continuously
differentiable along the real numbers, which, combined with the fact that it is
monotonically increasing means that a gradient can be calculated at any place,
facilitating "learning" as discussed in the section below. The specifics of the
ELU compared to its variants such as the rectified linear unit (ReLU) and leaky
ReLU, are too specific
to address here. It has been shown, however, to speed up learning on deep networks
as well as lead to better classification accuracy [@Clevert_et_al_2015].

### Artificial Learning

Neural networks with at least a single hidden layer can be shown to approximate
any continuous function [@Hornik_1991]. NNs achieve this impressive result by "learning"
the appropriate weights and biases, progressively updating these values until
an approximation is sufficiently close. There are many different algorithms for
updating the weights ($w$) and biases ($b$) on the networks
signals. In the interest of brevity, we
focus on the most widely used learning rule, Stochastic Gradient Descent (SGD)
with backpropagation. For an introduction
to additional rules such as the Widrow-Hoff, Generalized Delta, and Error-Correction
learning rules, see [@Engelbrecht_2007].

#### Stochastic Gradient Descent

Consider a fully connected 3 layer feed forward neural net,
depicted in Figure \ref{feedforward_net}. Each neuron in the $i^{th}$ layer is
connected to each neuron in layer $i + 1$ and inputs are passed forward
(that is, right to left)
through the network.

Stochastic gradient descent attempts to minimize the value of an error
function (also known as an optimization or cost function), $\mathcal{C}(y - y')$,
by progressively updating the network weights ($w$) and biases ($b$) in such a
way as to follow the first derivative gradient of $\mathcal{C}$ with respect to
the weights and biases ($\frac{\partial{\mathcal{C}}}{\partial{w}}$,
$\frac{\partial{\mathcal{C}}}{\partial{b}}$). By
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
The algorithm has two general steps:

1.  Inputs are passed through the network and output values calculated, providing
    the values needed to calculate epsilon and the derived gradients.
2.  The error signal from the output layer is propagated backwards through the
    network such that each weight and bias is updated according to its effect on
    the overall error of the cost function.

Step two, above is achieved by calculating the gradient of the cost function ($C$)
with respect to each weight and bias. These values are obtained by first obtaining
the network error at the output layer and backpropagating that error through the
preceding layers such that the contribution of each edge's weights and biases
can be calculated.

Consider a network of $L$ layers, with a cost function $C$, and an activation
function $\sigma$. Then, the vector of errors at the output layer is given in
Equation \eqref{eq:bp1}.

\begin{equation} \label{eq:bp1}
    \delta^{L} = \nabla_a C \odot \sigma'(z^L)
\end{equation}

where $\nabla_a C$ is vector of partial derivatives of the cost function with
respect to the observed output values at each neuron ($\frac{\partial C}{\partial a_j^L}$),
$\odot$ is the Hadamard product, and $z^L$ is the vector of inputs to all neurons
in the $L^{th}$ layer.

Using the error in layer $l + 1$ and the weight matrix, the error in the $l^{th}$ layer
can be calculated.

\begin{equation} \label{eq:bp2}
    \delta^{l} = ((w^{l+1})^T \delta^{l + 1})) \odot \sigma'(z^L)
\end{equation}

where $w^{l}$ is the weight matrix for the edges input to the $l^{th}$ layer of
the network. The rate of change of the cost function with respect to the bias
is exactly equal to the error given by Equation \eqref{eq:bp3}:

\begin{equation} \label{eq:bp3}
    \frac{\partial C}{\partial b_j^l} = \delta^l_j
\end{equation}

where $j$ refers to the $j^{th}$ neuron in the layer $l$. Finally, the contribution
of the network weights to the derivative of $C$ is simply the error weighted
by the previous layer's activation:

\begin{equation} \label{eq:bp4}
    \frac{\partial C}{\partial w^l_{jk}} = a^{l - 1}_k \delta^l_j
\end{equation}

where $w^l_{jk}$ is the $k^{th}$ edge weight into the $j^{th}$ neuron in the
$l^{th}$ layer.

Equations \eqref{eq:bp3} and \eqref{eq:bp4} are used as the input to gradient descent
which updates the weights accordingly.


## Convolutional Autoencoders

Consider a neural network tasked with learning relevant features
from an input dataset of $k$, $m \times n \times p$ images. Intuitively, a
network might seem
like an appropriate tool for identifying
relationships between pixels: if pixel values are passed to neurons, then
groups of pixels can be modeled by connected groups of neurons. If the weights
connecting neurons are allowed to vary over time, as is characteristic of
NNs, then it seems plausible that the network could model image structure via
neuronal relationships. Convolutional neural nets build on this intuition
by defining multiple layers that attempt to detect different abstractions, and
which contain multiple feature filters in each layer, each filter
targeting a different pixel structure.

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

Consider the $4 \times 4$ square input image depicted in Figure \ref{four_by_four}

![$4 \times 4 \times 1$ image.\label{four_by_four}](img/four_by_four.png){ width=60% }

In a fully connected network, each pixel value would be input to each neuron in the
first hidden layer. Conceptually, this means that any given neuron in the first
hidden layer receives information from every pixel in the image, specifically
the pixel value transformed by the weight and bias of the respective connection.
For the image context, it intuitively seems that this structure is non-optimal:
the relationship between the four pixel values of an image's corners seems
irrelevant compared to the relationship of the 4 pixels in any $2 \time 2$ region
of the image. The pixel relationships in the latter seems much more relevant to
a model concerned with recognizing image structure at some abstraction layer.
Convolutional neural networks acknowledge this distinction and differ from
fully connected networks by defining a local receptive
field of size $h$, where ($h \times h$) is the number of adjacent inputs that will connect to
the $j^{th}$ neuron in layer $i$.

![$4 \times 4 \times 1$ image with a $2 \times 2$ receptive field outputting to a $3 \times 3$ layer.\label{receptive_field}](img/four_by_four_receptive_field.png){ width=60% }

For each neuron in the hidden layer, the local receptive field is shifted adjacently
by a stride length of $l$ pixels. For example, in Figure \ref{receptive_field},
a stride length of $2$ corresponds to a total
of $9$ local receptive fields, resulting in a $3 \times 3$ hidden layer.
By limiting the number of input signals passed to a single neuron, each neuron
receives information from only adjacent pixels.

Furthermore, by design, each neuron in the $i^{th}$ hidden layer has the same weights
and biases. So, in Figure \ref{receptive_field}, the $2 \times 2$
array of weights and biases input to the
$j^{th}$ neuron in the $i^{th}$ layer are identical to every other array of weights and biases
in the $i^{th}$ layer. By making this restriction, the network ensures that all the neurons
in the $i^{th}$ layer are detecting the same spatial structure. This map from an input
layer to a hidden layer is called a feature map, and the set of weights and biases
that define a feature map is called a filter or a kernel [@LeCun_et_al_1990].
By increasing the number of feature maps at each layer, the network is able to
detect multiple features [@Bengio_et_al_2013].

CNNs aggregate the activations of the convolution layers
in what are known as pooling layers. By pooling adjacent activations, typically by taking
the maximum activation from a $l \times l$ sized area, the feature maps undergo
dimension reduction. Intuitively, this can be thought of as a function which outputs
whether a certain feature is found anywhere within a subsection of a layer.

The combined effect of these three components; (1) limiting network
connectedness, (2) sharing weights within layers, and (3) dimension reduction via
pooling layers; is a network which
builds a set of feature detectors while simultaneously reducing dimensionality of
the input image, resulting in a vector containing information on the spatial
structure of the input image.

### Autoencoders

Autoencoders are a type of unsupervised artificial neural network consisting of
two components, an encoder, $x' = f(x)$,
and a decoder $g(x')$, which perform, respectively, dimension reduction and
expansion in such a way as to minimize the error of
$y - y' = y - g(f(x))$, where $x'$ is a projection of the input, $x$, from the
data space, into
a much lower dimensional latent space. Figure \ref{ff_autoencoder} shows a
rudimentary autoencoder.

![A simple feedforward autoencoder\label{ff_autoencoder}](img/autoencoder.png){ width=100% }

Autoencoders have proved especially useful in denoising and dimension reduction
of images, which are naturally highly dimensional. Typically the reconstruction
loss function is taken to be the mean squared error:

\begin{equation} \label{eq:mse}
L = \frac{1}{n}\sum_{i=1}^{n} (x'_i - x_i)^2 = \frac{1}{n}\sum_{i=1}^{n} (g(f(x)) - x_i)^2
\end{equation}

The result is that the autoencoder encodes the image in the much reduced
feature space, similar to a nonlinear version of Principle Component Analysis
(PCA).

The specifics of the dimension reduction component of the encoder varies based on
specific architectures. Convolutional autoencoders, which we implement in this work,
are autoencoders that choose convolutional neural nets as encoders and decoders.
When optimizing for a convolutional autoencoder, the weights of the encoder are
updated in the direction which results in the largest decrease in image
reconstruction loss. The combined effect is a network which defines a dimensionality
reduction function (autoencoder) where the output space is optimized for the image features
that are most relevant for reconstructing the input image (CNN).

## DCEC-Paint

In this work, we evaluate the clustering performance of the Deep Convolutional
Embedded Clustering (DCEC-Paint) algorithm on a new dataset of digitized fine
art. This network is
identical to the one specified in [@Castellano_and_Vessio_2020], who made minor
adaptations to the DCEC algorithm specified by [@Guo_et_al_2017]. Figure \ref{arch}
shows the architecture of the network. It is convolutional autoencoder with a
deep learning clustering algorithm fed from the latent feature space.

![Deep Convolutional Embedded Clustering (DCEC-Paint) architecture (adapted from [@Guo_et_al_2017]).\label{arch}](img/network.png){ width=100% }

Unlike previous deep clustering algorithms, DCEC-Paint attempts to maintain feature
space integrity by jointly optimizing for reconstruction and clustering loss.
Previous algorithms only optimize for clustering after the network is pre-trained
which, over the course of clustering can result in divergence in the feature space
from the pre-trained spatial structure. We posit
that retaining learned features is especially important for clustering digitized
artworks which are, on the whole, spatially complex, and should in theory store a
large amount of artistically meaningful data in the feature space.

As mentioned above, the network is tasked with minimizing the overall loss, a
combination of the image reconstruction and clustering loss:

\begin{equation} \label{eq:loss}
L = \gamma L_r + (1 - \gamma) L_c
\end{equation}

where $\gamma$ is a tunable parameter. [@Guo_et_al_2017] set $\gamma = 0.1$,
while [@Castellano_and_Vessio_2020] used $\gamma = 0.9$, putting more importance
on optimizing for clustering loss rather than reconstruction loss. The goal of
clustering artwork is to optimize cluster assignment, so we follow [@Castellano_and_Vessio_2020]
and set $\gamma = 0.9$.

Below we describe the derivation of the reconstruction and clustering loss in
detail, with their component parts.

### Autoencoder

As shown in Figure \ref{arch}, the encoder expects a $128 \times 128$ RGB image with pixel values
scaled between $0$ and $1$. The encoder consists of three convolutional layers which
have $32$, $64$, and $128$ filters, respectively. In all cases the stride length is $2$ pixels
and the kernel size (local receptive field) is $5 \times 5$ for the first two convolutional
layers, and $3 \times 3$ for the final layer. All layers use the ELU activation function.
The output of the final convolutional layer is flattened in to a vector of size
$327,684$, which is fully connected to the embedded space. We follow
[@Castellano_and_Vessio_2020] and set the size of the embedded space to $32$.
From the embedded space, the decoder upsamples images with an architecture that
mirrors the encoder.

### Clustering

The architecture of the clustering layer is derived from [@Xie_et_al_2017], who
introduced the method as part of Deep Embedded Clustering (DEC) which, after
pre-training a feature space with stacked autoencoders, optimizes network
parameters by computing a probability distribution for membership to
a set of cluster centroids and uses stochastic gradient descent via
backpropagation to learn an mapping which minimizes the Kullback-Leibler
(KL) divergence to that distribution.

DCEC uses the same principles, substituting, as mentioned above, a CAE for stacked
autoencoders, and retaining the CAE during the final learning step. After pre-training
the network to learn an initial feature space (i.e. $\gamma = 0$ in Equation \eqref{eq:loss}),
we set the cluster centroids to
initial values using K-means and set $\gamma = 0.9$. DCEC then iteratively
performs two steps to learn cluster weights:

1.  Computes the similarity between a point in the feature space, $z_i$, and a centroid,
    $\mu_j$ using Student's t-distribution, interpreted as the probability that
    sample $i$ belongs to cluster $j$.
2.  Calculates the clustering loss via KL divergence and updates the target distribution
    for future assignments.

We expand on these two steps below.

#### Calculation of Cluster Probability

The probability that a sample from the feature space, $z_i$, belongs to cluster $j$
is taken to be given by Student's t-distribution with one degree of freedom
(i.e. a Cauchy distribution), shown in Equation \eqref{eq:cauchy}.

\begin{equation} \label{eq:cauchy}
q_{ij} = \frac{(1 + (z_i - \mu_j)^2)^{-1}}{\sum_{j'}(1 + (z_i - \mu_{j'})^2)^{-1}}
\end{equation}

[@Xie_et_al_2017] uses a t-distribution after [@Maaten_et_al_2008], who introduced
it for t-SNE, a technique for visualizing high dimensional spaces in two or three
dimensions. Thus, $q_{ij}$ can be interpreted as the probability that sample $i$
belongs to cluster $j$, under the assumption that the data follow a t-distribution.

#### Minimizing Clustering Loss

<!-- https://stats.stackexchange.com/questions/387500/clustering-with-kl-divergence -->

<!-- TODO: the only thing this sentence says is that you don't understand this math -->
To improve the cluster centroids, DEC attempts to compare the observed cluster
assignment probability distribution ($q_{ij}$ in \eqref{eq:cauchy}) to
a target distribution, $p_i$, by measuring the KL
divergence. KL divergence attempts to measure the similarity of two distributions
and, in this context, can be thought of as the amount of information lost when
using $q_i$ to approximate $p_i$. Equation \eqref{eq:kl} gives the KL divergence
definition.

\begin{equation} \label{eq:kl}
L = KL(P||Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}
\end{equation}

$L$ is the network's clustering loss, which is used in combination with reconstruction
loss to form the network's overall loss, given by Equation \eqref{eq:loss}.

$p_i$ is chosen in [@Xie_et_al_2017] to be the squared $q_i$, normalized by the
cluster size:

\begin{equation} \label{eq:aux}
p_{ij} = \frac{q_{ij}^2/f_j}{\sum_{j'} g_{ij'}^2/f_{j'}}
\end{equation}

where $f_j = \sum_i q_{ij}$, the frequency of per cluster.
By squaring the observed distibution, the probability mass of $p_i$ is more concentrated
about the centroid, forming purer clusters than $q_i$, thus creating a divergence
between the distributions that facilitates learning.
$p_i$ has favorable
properties for learning, which [@Xie_et_al_2017] outline as: (1) it's use in
Equation \eqref{eq:kl} will lead to more distinct clusters; (2) data points with
high confidence contribute more to the distribution; (3) each cluster centroid's
contribution to the KL divergence is normalized by the cluster size so that large
clusters do not out compete smaller ones. [@Xie_et_al_2017] used empirical results
to show that $p_i$ exhibits these properties.

The overall effect of the clustering loss function is that samples
which have a high confidence in belonging to certain cluster will contribute
largely to defining the centroids for the target distribution, $p_i$.
As a consequnce, the gradient of $L$ points more in the direction of these canonical
examples, and network weights and biases are updated in a way that builds
a feature space which with purer clusters.

#### Optimization

For a given number of clusters, $k$, the full DCEC-Paint algorithm proceeds as follows:

1. Set $\gamma = 1$ in Equation \eqref{eq:loss}, and pretrain the network for a
   number of epochs to build the feature space. This is equivalent to training
   a vanilla convolutional autoencoder.
2. Initialize $k$ cluster centroids using K-means.
3. Set $\gamma = 0.1$ in Equation \eqref{eq:loss}. Define a mini-batch size $b$,
   and while the stopping criteria is not been met, for each mini-batch:
    1. Calculate $L_c$ and $L_r$ and update the autoencoder weights and cluster
       centers according to SGD via backpopagation using
       $\frac{\partial L}{\partial z_i}$ and
       $\frac{\partial L_c}{\partial \mu_i}$.
    2. On every $T^{th}$ iteration, where $T$ is an update interval, update the
       target distribution $p$ using the entire dataset's most recent
       predictions.


The algorithm terminates when after the proportion of examples that change clusters
between two mini-batches is less than some tolerance $\delta$.

<!-- #### Prediction, Optimization, Parameter Initialization, etc. -->
# Methods

## Dataset

Christie's hosts the results online of all public auctions held
after December, 2005. This amounts to a truly impressive record of fine art,
furniture, antiques, jewelry, and other collectible sales; well over half a
million pieces auctioned over hundreds of sales held across the world and
online. As one of world's two leading auction houses (the other being Sotheby's),
these auction results contain thousands of influential and expensive artworks,
including the most expensive piece ever sold, da Vinci's Salvator Mundi[^mundi], which
sold for an astounding $450 million.

[^mundi]: https://www.christies.com/lotfinder/paintings/leonardo-da-vinci-salvator-mundi-6110563-details.aspx

To obtain the final dataset used in this work, we scraped the Christie's
website for all auctions
that consisted primarily of artwork in two dimensional mediums. Left with a
set of nearly $300$ thousand works of art, we chose the top $50$ most prolific
artists and,
randomly selecting $250$ works of art for each artist, we hand curated the dataset
to exclude intermediate sketches and sculptures. The final dataset contains
$n = 10,505$ $3$ channel (RGB) $128 \times 128$ pixel images which span a
diverse set of artists, mediums, and movements. The full set of artists along
with their respective mediums and movements can be seen in
table \ref{tab:artists}.

Compared to previous applications of this algorithm, we believe this dataset
to represent a much broader set of artists, genres, and mediums. The dataset
includes over a thousand photographs, including varied works by
Ansel Adams, Henri Cartier-Bresson, and Hiroshi Sugimoto. The majority of the
dataset consists of a diverse set of paintings including 19th century
Traditional Chinese paintings, Rennaissance works, and Pop Art. Notably, the
dataset includes a large body of modern and contemporary works.

With an artistically diverse dataset, we expect the algorithm to be particularly
exposed to the cross depiction problem. Imagine two digitized works depicting
a clock but of different artistic styles and mediums. One, Salvador Dali's famous
1931 "Persistence of Memory", and the other a moment captured by the
lens of Henri Cartier-Bresson. Holding all else equal, we expect a "correct"
clustering algorithm to find these two works more similar than a separate pair
of works that don't contain clocks. The algorithm's ability
on these qualitative tasks is additionally important to the quantitative
measures outlined below.

<!-- TODO - You could include a plot with distribution of image prices -->
<!-- Maybe a figure would be better here? -->
\begin{singlespace}
\begin{table}[]
\centering
\footnotesize
\begin{tabular}{p{4cm}p{0.5cm}p{0.5cm}p{0.5cm}p{3.5cm}p{4.5cm}}
\hline
Artist                       & n   & Birth & Death & Mediums                     & Movement                                         \\ \hline
henri cartier-bresson        & 250 & 1908  & 2004  & photography                 & street photography                               \\
victor vasarely              & 250 & 1906  & 1997  & painting,sculpture          & Op Art                                           \\
ansel adams                  & 249 & 1902  & 1984  & photography                 & Group f/64                                       \\
sam francis                  & 249 & 1923  & 1994  & painting,printmaking        &                                                  \\
hiroshi sugimoto             & 249 & 1948  &       & photography                 &                                                  \\
maurice de vlaminck          & 246 & 1876  & 1958  & painting                    & fauvism                                          \\
wayne thiebaud               & 246 & 1920  &       & painting                    & pop art,new realism,bay area figurative movement \\
karel appel                  & 244 & 1921  & 2006  & painting,drawing,sculpture  & cobra                                            \\
andy warhol                  & 243 & 1928  & 1987  & print making,painting       & pop art                                          \\
bernard buffet               & 242 & 1928  & 1999  & panting,drawing,printmaking & expressionism                                    \\
keith haring                 & 242 & 1958  & 1990  & painting,drawing            & pop art,street art                               \\
gerhard richter              & 241 & 1932  &       & photography,painting        & Capitalist realism                               \\
jasper johns                 & 239 & 1930  &       & painting,printmaking        &                                                  \\
alighiero boetti             & 238 & 1940  & 1994  & painting                    & Abstract expressionism, Neo-Dada, pop art        \\
robert motherwell            & 238 & 1915  & 1991  & painting,printmaking        & Abstract expressionism                           \\
marc chagall                 & 237 & 1887  & 1985  & painting                    & cubism,expressionism                             \\
helmut newton                & 236 & 1920  & 2004  & photography                 &                                                  \\
jean dubuffet                & 236 & 1901  & 1985  & painting                    & Art Brut                                         \\
irving penn                  & 235 & 1917  & 2009  & photography                 &                                                  \\
robert rauschenberg          & 234 & 1925  & 2008  & painting                    & neo-dada,abstract expressionism                  \\
jim dine                     & 233 & 1935  &       & painting                    & neo-data,pop art                                 \\
joan miro                    & 233 & 1893  &       & painting                    & surrealism,dada,experimental                     \\
frank stella                 & 233 & 1936  &       & painting,printmaking        & modernism,abstract expressionism                 \\
christo                      & 232 & 1935  &       & painting                    & Nouveau réalisme                                 \\
tom wesselmann               & 232 & 1931  & 1945  & painting                    & pop art                                          \\
takashi murakami             & 230 & 1962  &       & contemporary art            & superflat                                        \\
roy lichtenstein             & 229 & 1923  & 1997  & painting                    & pop art                                          \\
sol lewitt                   & 229 & 1928  & 2007  & painting,drawing            & conceptual art,minimalism                        \\
zao wou-ki                   & 228 & 1920  & 2013  & painting                    &                                                  \\
damien hirst                 & 227 & 1965  &       & painting                    & young british artists                            \\
raoul dufy                   & 226 & 1877  & 1953  & painting                    & fauvism,impressionism,modernism,cubism           \\
qi baishi                    & 222 & 1864  & 1957  & painting                    & guohuo                                           \\
david hockney                & 213 & 1937  &       & panting,print making        & pop art                                          \\
zhang daqian                 & 210 & 1899  & 1983  & painting                    & guohuo                                           \\
laurence stephen lowry, r.a. & 209 & 1887  & 1976  & painting                    &                                                  \\
pierre-auguste renoir        & 203 & 1841  & 1919  & painting                    & impressionism                                    \\
alexander calder             & 202 & 1898  & 1976  & sculpture                   &                                                  \\
francis newton souza         & 199 & 1924  & 2002  & painting,drawing            & progressive art                                  \\
max ernst                    & 198 & 1891  & 1976  & painting                    & dada,surealism                                   \\
albrecht dürer               & 191 & 1471  & 1528  & painting,printmaking        & high renaissance                                 \\
pu ru                        & 179 & 1896  & 1963  & painting                    & guohuo                                           \\
lucio fontana                & 174 & 1899  & 1968  & painting                    & spatialism                                       \\
salvador dalí                & 173 & 1904  & 1989  & painting                    & cubism,dada,surrealism                           \\
rembrandt harmensz. van rijn & 171 & 1606  & 1669  & painting                    & dutch golden age,baroque                         \\
henry moore                  & 148 & 1898  & 1986  & bronze scuplture            & modernsim                                        \\
henri de toulouse-lautrec    & 120 & 1864  & 1901  & painting                    & post-impressionism,Art Noveau                    \\
pablo picasso                & 117 & 1881  & 1973  & painting                    & cubism,surrealism                                \\
henri matisse                & 99  & 1869  & 1954  & painting                    & fauvism,modernism,post-impressionism             \\
edgar degas                  & 83  & 1834  & 1917  & painting                    & impressionism                                    \\
auguste rodin                & 18  & 1840  & 1917  & drawing                     &                                                 
\end{tabular}
\caption{The 50 artists included in this dataset ($n$ - number of images included in dataset)}
\label{tab:artists}
\end{table}
\end{singlespace}


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

We re-implemented DCEC on the Tensorflow v1.5 [@tf_2015] framework and trained it on
Google Kubernetes Engine with a single NVIDIA Tesla T4 GPU with 16 GB of GDDR6.
Following [@Castellano_and_Vessio_2020], we used an update interval of $140$, a
learning rate of $0.001$, and pre-trained for $200$ epochs before attaching the
clustering layer. With a larger GPU than the previous authors used, we increased
batch size from $256$ to $512$. The model took on the order of 1-2 hours to
cluster after pretraining the convolutional autoencoder. Clustering terminated
when the proportion of samples which changed clusters between two consecutive
update intervals was less than $0.001$. We ran the network for $k \in \{1..10\}$.

<!-- # Prediction Evaluation -->
<!-- How did we evaluate the model's performance -->
<!-- Comparison to the estimate for the painting. Those are the experts -->
<!-- Comparison to alternative methods? What would those be? Is there precedence? -->
<!-- Color Pallete, smoothness, brightness, and portrait scores -->

# Clustering Evaluation
<!-- How did we evaluate the performance of the network? -->

While this dataset contains metadata, the digitized works included in this
study are unlabeled in the sense that there is no "correct" cluster for each image.
As such, we evaluate model
performance with two global methods, the average silhouette score and
the Calinski-Harabasz index [@Calinski_Harabasz_1974]. We also measure the
GAP statistic [@Tibshirani_2000]
on the unclustered embedded space obtained after running the algorithm for $k=1$,
which the equivalent of pre-training the CAE to completion.
This indicates the presense/absence of clusters before the algorithm begins
learning cluster centers in earnest. In all cases we use euclidean distances.

The silhouette score measures how similar a data point is to its own cluster,
relative to the nearest cluster and is given, for a single datatum $x_i$, as:

\begin{equation} \label{eq:ss}
s_i = \frac{b_i - a_i}{max\{a_i, b_i\}}
\end{equation}

where $a_i$ and $b_i$ are the mean distances from $x_i$ to all points in the same
cluster and the nearest cluster, respectively. The Silhouette score ranges from
$-1$ to $1$, where $1$ indicates better defined clusters. In our evaluation we
use the average silhouette score over all datapoints as a measure of clustering
performance.

The Calinski-Harabasz index measures the ratio of between cluster dispersion and
within cluster dispersion according to the following:

\begin{equation} \label{eq:ch}
CH_k = \frac{SS_B}{SS_W} \times \frac{N - 1}{k - 1}
\end{equation}

where $SS_W$ and $SS_B$ are the within and between cluster dispersion, given as:

\begin{equation} \label{eq:chssb}
SS_B = \sum_{j=1}^{k} n_j ||m_j - m ||^2
\end{equation}
and
\begin{equation} \label{eq:chssw}
SS_W = \sum_{j=1}^{k} \sum_{x \in c_j} ||x - m_j ||^2
\end{equation}

Where $k$ is the number of clusters, $n_j$ is the number of datapoints in the
$j^{th}$ cluster, $m_j$ is the centroid of the $j^{th}$ cluster, $m$ is the
mean of all the data, and x is the given datapoint.

The Calinski-Harabasz index is unbounded in the positive direction. For this
reason, we normalize the results for each cluster solution ($k \in {1...10}$) to
the largest value to enable comparison. Higher values indicate a lower relative
within cluster dispersion, i.e. tighter
clusters.

DCEC and its predecessor algorithms do not extend to the one/no cluster solution.
At each iteration in the clustering algorithm, the probability that sample $i$
belongs to cluster $k$ is calculated according to Equation \ref{eq:cauchy}. In
the once cluster solution, this probability is $1$ and does not change,
leading to a Kullback-Leiber divergence of zero. This implies that the model
optimizes exclusively for image reconstruction loss. In other words, the
algorithm is reduced to a CAE.

This case is interesting, however, as it gives insight into the state of the
algorithm after pretraining and highlights the importance of the clustering
component in iteratively learning a feature space of $k$ clusters. To evaluate
the presense of clusters after pretraining, we report the GAP
statistic [@Tibshirani_2000]. This statistic, for a given number of clusters,
$k$, examines the within cluster dispersion relative to the expected value of
some appropriate null reference distribution for the within cluster dispersion.
The appropriate cluster choice is the value for which the observed within cluster
distribution is closest to the expected value:

\begin{equation} \label{eq:gap}
GAP_k = E(log(SS_{W_k})) - log(SS_{W_k})
\end{equation}

Where $E$ is the expected value, and $SS_W$ is given by equation \ref{eq:chssw}.
For a complete derivation, see [@Tibshirani_2000].

In addition to these quantitative metrics we perform a more qualitative assesment
of the results for the top performing clusters. In particular we examine (1) the
distribution of a few artists works across clusters; (2) the overall interpretation
of the two top perfoming clusters; and (3) the distribution of different artistic
styles across the clusters.

# Experiment Results
<!-- What did we see? -->

Table \ref{cluster_scores} and Figure \ref{score_plot} show the average
silhouette and Calinski-Harabasz scores for values of k between $2$ and $10$,
before and after beginning the clustering component of the algorithm (CAE+Kmeans
vs DCEC, respectively).
The average silhouette and Caliski-Harabasz scores disagree slightly on the
optimal cluster solution. Using silhouette scores suggests that the optimal
number of clusters is 8 (followed by 9 and 10), while the relative CH
coefficient suggests that the optimal solution is 3, followed by 8 and 9.
For all values of $k$, DCEC was clearly effective in clustering the data as the
within to between cluster dispersion values are much higher for DCEC than the
CAE+Kmeans solution. This is expected as the algorithm is learning a feature
set that partially optimizes for clustering loss.

\begin{singlespace}
\begin{table}[]
\centering
\begin{tabular}{lllll}
\hline
k  & ss - CAE+Kmeans & ch - CAE+Kmeans & ss - DCEC & ch - DCEC \\ \hline
2  & 0.2090          & 1.0000          & 0.7929    & 0.4744    \\
3  & 0.1426          & 0.8219          & 0.8280    & 1.0000    \\
4  & 0.1028          & 0.5754          & 0.7750    & 0.4775    \\
5  & 0.0934          & 0.4496          & 0.8073    & 0.5089    \\
6  & 0.0819          & 0.3705          & 0.7834    & 0.3245    \\
7  & 0.0884          & 0.3784          & 0.7863    & 0.3873    \\
8  & 0.0811          & 0.3301          & 0.8702    & 0.9098    \\
9  & 0.0463          & 0.2369          & 0.8520    & 0.6518    \\
10 & 0.0772          & 0.2750          & 0.8284    & 0.4480   
\end{tabular}
\caption{Average silhouette and Calinski-Harabasz scores by cluster}
\label{cluster_scores}
\end{table}
\end{singlespace}

![Average Silhouette and Calinski-Harabasz scores across number of clusters ($k$). According to CH, the optimal cluster solution is $3$, while silhouette scores suggest the optimal solution is $8$. For all values of $k$ the algorithm achieves a high ratio of within to between cluster dispersion.\label{score_plot}](/home/dubs/dev/paap/img/dcec_metrics.png){ width=100% }

It's clear that Kmeans, which is initally used to set the cluster centroids prior
to learning the final feature space, is quite ineffective in producing defined
clusters. We expect this, to some extent, as the pretraining autoencoder step
is exclusively optimized to minimize reconstruction loss. We examined the pre-clustering data by
running the Autoencoder for *20,000* epochs.
Figure \ref{gap} plots the GAP statistic for a range of cluster sizes. The gap statistic increases throughout
the range, indicating there is no clear "correct" number of clusters for the
embedded space only - a single (zero) cluster is the optimal solution. This
highlights the importance of the clustering component in learning the feature
space.

![GAP statistic for cluster sizes $k=1$ through $k=20$, performed on the embedded space after running the autoencoder, without an attached clustering layer, for *20,000* epochs. This figure demonstrates the absence of distinct clusters in the pretrained dataset. See text for discussion.\label{gap}](/home/dubs/dev/paap/img/1/gap.png "gap"){ width=100% }


Figure \ref{all_tsne} shows all t-SNE plots for $k \in \{2..10\}$ for the final
32-dimensional feature space. For all values of $k$ the algorithm
forms readily discernable clusters, as depicted by t-SNE.

![t-SNE diagrams of the final $32$ dimensional feature (embedded) space. For all values of $k$ the algorithm appears to build a distinctly partitioned feature space.\label{all_tsne}](/home/dubs/dev/paap/img/all_tsne.png){ width=90% }

Figure \ref{tsne_evolution} visualizes cluster evolution. Immediately after the
pretraining step (CAE+Kmeans) there are no discernable clusters. Within a few
thousand epochs, however, the algorithm has learned a feature space which shows
distinct clusters.

![Cluster evolution for $k=8$. Images are t-SNE projections at the following iterations: 0, 612, 1224, 1836, 2448, 8262 (final epoch).\label{tsne_evolution}](/home/dubs/dev/paap/img/8/cluster_evolution.png){ width=100% }

Diving deeper into the individual cluster results, Figure \ref{images_8} shows
a selection of images for each of the clusters in the $8$ cluster solution.
Figure \ref{images_3} shows a selction for the $k=3$ solution. In these examples,
the within cluster samples appear to be visually cohesive, but it's not immediately
clear what artistic features each cluster share. These samples suggest that the
clustering does not strictly follow artist or genre. For instance, in Figure
\ref{images_8}, the $2^{nd}$ and $6^{th}$ clusters both contain works by L.S.
Lowry, an English artist known for his drawings and paintings of industrial
scenes of Northwest England. Indeed, both clusters show the characteristic Lowry
work, with drawings of public outdoor spaces populated with bustling figures,
set against a backdrop of in-use industrial buildings. L.S. Lowry's collective
body of works narrowly tracks a distinct style and feel, so it's interesting
that his work is crosses cluster boundaries.

![A selection of images from the $k=8$ cluster solution\label{images_8}](img/8/collage.png){ width=100% }

![A selection of images from the $k=3$ cluster solution\label{images_3}](img/3/collage.png){ width=100% }

Figures \ref{tsne_artists} and \ref{tsne_genre} show the distribution of a set
of artists and genres throughout the clusters, respectively. It's apparent from
these figures that the clustering does not (strictly) follow artist or genre
differences as all categories appear to be distributed across two or more
clusters, for the $8$ clsuter solution.

![A selection of artists artwork against the $k=8$ cluster solution\label{tsne_artists}](img/8/tsne_artists.png){ width=50% }

![A selection of artists artwork against the $k=3$ cluster solution\label{tsne_artists}](img/3/tsne_artists.png){ width=50% }

![A selection of genres against the $k=8$ cluster solution\label{tsne_genre}](img/8/tsne_genre.png){ width=50% }

![A selection of genres against the $k=3$ cluster solution\label{tsne_genre}](img/3/tsne_genre.png){ width=50% }


Figure \ref{tsne_medium} shows the distribution of photographs vs non-photographs.
Like genre and artist, it's apparent that clustering appears to be medium agnostic.

![Photographs vs non-photographs for the $k=8$ cluster solution\label{tsne_medium}](img/8/tsne_photograph.png){ width=50% }

![Photographs vs non-photographs for the $k=3$ cluster solution\label{tsne_medium}](img/3/tsne_photograph.png){ width=50% }


Rather than proxying aesthetic differences between genre, artist, and medium, the
algorithm appears to be clustering on much more fundamental characteristics
of the images - e.g. busyness, the presence/absence of long lines, brush
stroke style, and color.


# Conclusion


# References

<!-- TODO
2. Calculate the distribution of red/green/blue in the clusters for the 3 and 8 cluster solutions
3. Look at grayscale images vs 3 channel for the 3 and 8 cluster solutions
4. Calculate some engineered features for each image, If we cluster on those,
   how much overlap is there? Openness, etc
5. Write the rest of the discussion
6. Write conclusion
7. What happens when we grayscale all the images and cluster on them?
8. What happens when we introduce vertical white lines into the images? Is the algorithm fragile?
 -->
