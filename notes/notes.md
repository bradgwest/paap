# Notes

Because I don't know anything about statistics/ML any more :(

## Englebrecht - Computational Intelligence

### Chapter 1 - Using Neural Nets to Recognize Handwritten Digits

>   The firing of an artificial neuron (AN), and the strength of the exiting signal
    are controlled via a function, referred to as the activation function. The
    AN collects all inncoming siganls and computes a net input signal as a
    fucntion of the respective weights.

>   A nn is a realization of a nonlinear mapping from R^I to R^K, where I and K
    are, respectively, the dimension of the input and target (desired output)
    space. The function that does this is usually a complex function of a set
    of nonlinear functions, one for each neuron in the network.

*   Epoch - One iteration of training in which we've exhausted the training
    samples

*   Stochastic gradient descent - A version of gradient descent in which the
    partial derivatives of the cost functions with respect to the weights and
    biases are updated for only a random sample of the inputs.

*   mini-batch - a batch of training items to train on before updating the
    weights and biases

*   sigmoid neuron a neuron that is not a perceptron because it uses a sigmoid
    activation function, namely sigma(z) = 1 / (1 + e^-z). Different activation
    functions use different values for the partial derivatives, which makes the
    network behave differently

*   Deep neural nets are networks that have two or more hidden layers. We can
    think about a neural network which is answering a really complicated
    question by breaking it down into really tiny, simple questions, like going
    from "is that a face" to "is that an eyebrow". Those tiny questions are at
    the level of a few pixels

### 2 - How the Backpropagation Algorithm Works

*   

### 3 - Improving the Way Neural Networks learn

*   Use Softmax when you want to interpret the output activations a probabilities.

*   you can think of validation data as a type of training data that helps us
    learn good hyper-parameters.

*   One of the best ways of reducing overfitting is to increase the size of the
    training data

*   Regularization - the effect of regularization is to make it so the network
    prefers to learn small weights, all other things being equal. Large weights
    will only be allowed if they considerably improve the first part of the cost
    equation.

*   Regularized networks are constrained to build relatively simple models based
    on patterns seen often in the training data, and are resistant to learning
    the peculiarities of the noise in the training data.

#### Keywords
    - cost function
    - activation function
    - biases
    - weights
    - sigma
    - regularization parameter - lambda
    - weight initializers
    - backprogagation
    - stochastic gradient descent
    - training, test, validation data
    - cross entropy cost
    - learning rate - n
    - mini batches

#### Hyperparameters
    - Learning rate (eta)
    - L2 regularization parameter (lambda)
    - Mini-batch size
    - variable learning rate

#### How to tune hyperparameters

> First, we estimate the threshold value for η at which the cost on the
training data immediately begins decreasing, instead of
oscillating or increasing. This estimate doesn’t need to be too accurate. You can estimate
the order of magnitude by starting with η=0.01. If the cost decreases during the first few
epochs, then you should successively try η = 0.1, 1.0, . . . until you find a value for η where
the cost oscillates or increases during the first few epochs. Alternately, if the cost oscillates
or increases during the first few epochs when η=0.01, then try η=0.001,0.0001,. . . until
you find a value for η where the cost decreases during the first few epochs. Following this
procedure will give us an order of magnitude estimate for the threshold value of η. You
may optionally refine your estimate, to pick out the largest value of η at which the cost
decreases during the first few epochs, say η=0.5 or η=0.2 (there’s no need for this to be
super-accurate). This gives us an estimate for the threshold value of η.

    - https://dl.acm.org/doi/10.5555/2188385.2188395 -- Random search for hyperparameters

### Chapter 6 - Learning to train Deep Networks

*   This means that all the neurons in the first hidden layer detect exactly the same feature3
,
just at different locations in the input image. To see why this makes sense, suppose the
weights and bias are such that the hidden neuron can pick out, say, a vertical edge in a
particular local receptive field. That ability is also likely to be useful at other places in the
image. And so it is useful to apply the same feature detector everywhere in the image. To put
it in slightly more abstract terms, convolutional networks are well adapted to the translation
invariance of images: move a picture of a cat (say) a little ways, and it’s still an image of a
cat4

* Recall that, as mentioned earlier, ImageNet contains images of varying resolution.
This poses a problem, since a neural network’s input layer is usually of a fixed size. KSH dealt
with this by rescaling each image so the shorter side had length 256. They then cropped
out a 256 × 256 area in the center of the rescaled image. Finally, KSH extracted random
224 × 224 subimages (and horizontal reflections) from the 256 × 256 images. They did this
random cropping as a way of expanding the training data, and thus reducing overfitting.
This is particularly helpful in a large network such as KSH’s. It was these 224 × 224 images
which were used as inputs to the network. In most cases the cropped image still contains the
main object from the uncropped image.

## DEC Paper

* Minimizing  the  Kullback-Leibler  (KL)  divergence  be-tween a data distribution and an embedded distribution hasbeen used for data visualization and dimensionality reduc-tion (van der Maaten & Hinton, 2008)

* We take inspiration from parametric t-SNE. Instead of min-imizing  KL  divergence  to  produce  an  embedding  that  isfaithful to distances in the original data space,  we definea centroid-based probability distribution and minimize itsKL divergence to an auxiliary target distribution to simul-taneously improve clustering assignment and feature repre-sentation.

## How is artwork apprasied

https://www.christies.com/features/How-is-an-artwork-appraised-10033-1.aspx?lid=1
