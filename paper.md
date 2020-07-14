# TODO
* Get DCEC-Paint clustering for photos and prints, with their changes:
    - ELU rather than ReLU
    - larger latent embedding space
    - Loss importance on clustering rather than decoder
    - ON GCP
        + Docker image
        + Deploy script
        + Logs to GCS or something like that if it's longer than 5 minutes to trained
        + Needs to be efficient because you're going to do this a lot
        + Config file?

* Rebuild network with fully connected prediction layer
    - What is your loss function?
    - How do you set this? They did it with Kmeans, what should you do it with?
        + Mean of the training set
        + Median of the training set
        + Something random from the training set?
        + 0?

* Train, tune, validate, test on 50k print/photo dataset
    - clean that dataset??
    - Hopefully no need to clean that dataset further

* Depending on results:
    * Train, tune, validate, test on large dataset

* Get basic statistics about datasets

# Deep Convolutional Autoencoder Prediction of Art Action Prices

## Abstract
<!-- TODO Write this after you finish the paper -->

## Introduction
<!-- This is where your state the motivation -->
<!-- contributions to the field -->

## Related Work
<!-- DCN efforts, specifically deep clustering, like DEC, DCEC, DCEC-Paint -->
<!-- TODO Efforts to quantify art prices, especially using extracted, not learned features -->

## Deep Convolutional Neural Networks
<!-- Motivation for what NNs offer in general -->
<!-- What do NN offer to image problems -->
<!-- What do they offer to this specific problem -->

### Neural Network Basics
<!-- How and why do NNs work -->

### Convolutional Autoencoders
<!-- What are convolutional NNs, and how to they build on traditional NN? -->

### DCEP-Paint
<!-- Specifics of this algorithm   -->

#### Prediction, Optimization, Parameter Initialization, etc.
<!-- Methods go here -->

## Experiments

### Datasets
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
