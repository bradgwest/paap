# TODO

## Research Question

CNN for predicting art auction prices. Based on DCEC, only rather
than a clustering layer, we put in a continuous output layer which
attempts to predict the log of the auction price (would log actually
be a good thing to use here?). The result is that we actually learn
common features of the data with the autoencoder/decoder, and don't
throw that information away. To our knowledge this would be an original
contribution to the field.

Does an artwork's intrinsic artistic characteristics (the characteristics that
are not specific to things like it's provenance, creator, age, but are intrisic
to the work itself like shapes, scenes, and colors) contribute meaningfully to
that piece's sale price, after controlling for art movement/genre?

To answer this question we analyze the sale of thousands of pieces of
two-dimensional art which were sold at christie's auction house. We extract
the intrinsic characteristics of each piece, as well as the non-intrinsic
characteristics. Using proxies for artwork characteristics, we cluster the
results, pitting similar pieces next to similar pieces, and develop a
similarity score for the artwork.

We then fit various models using all terms to determine which characteristics
affect the price of the piece and determine if the artistic components is one
of that.

## Steps

## Data Cleaning
* Scale the images
    - Lombardi thesis talks briefly about size correction
    - Lee and Cha talk about resizing to the smallest dimension and then cropping to standardize
    - Going to scale the images to have a minimum dimension of 256. Then going to fit a functional convolutional neural net to handle variable image size
    - Castellano just says "scale"
    - Maybe only have images with a certain aspect ratio, then crop so that they're of a certain size? Do both?
    - Maybe scale to within a certain aspect ratio, then use the tensorflow lib to resize: https://www.tensorflow.org/api_docs/python/tf/image/resize
* X -Upload all images to gcs
* Filter local images to just include 2d pieces, or perhaps just include paintings
* X -Combine exchange rates with the general artwork dataset
* Add an is\_2d column to the dataset
* X - All small and relevant images locally
* X - Document the pipeline process to get images, and format them
* How do we find which images to process? What is a good size?
    - Randomly sample and see how many are non-2d. Based off of this, decide if
      we want to do a manual subset, random manual selection, somthing else

### Reading/Research
* Read about fitting a model for movement, can we do this easily? If so, then do it
* Read about clustering
* Read about feature extraction
* Read about artwork similarity
* Read about determining similarity
* Read about how to measure how strong similarity is
* Read about using similarity measures in a model (Maybe which cluster it's in?)

### Feature extraction
*   Feature extraction on the images

### Modeling
*   Can we get art movement/genre with a model? It would be really nice to
    control for this, or at least super impose it on some visualizations
*   Clustering the images - Decide if you're going to use pytorch, tensorflow,
    a different library
*   How do we evaluate a clustering model for efficacy?
*   Get a similarity measure for two pieces of artwork (This might not be necesary)
*   Fit models for an artwork's price (these will be log models using the cluster it's in)

### Visualization
*   Make visualization template so that all images are identifably the same
*   t-SNE for visualization
*   Overlaying two clustering models?

### Writing
*   Find markdown workflow to write in, with bibtex
*   Read about best way to write an academic paper
*   Write the paper

### MSU
*   Email Mark, Katie, and Andy with the paper and repo, asking if it, with
    some work, is good enough
*   Call MSU and ask them what needs to be done to finish
        - Can I take some remote classes for the 2 extra credits?
        - Can I do an independent study for the extra credit?
        - What do I need to do to not be coming to the school?
        - What happens if I don't finish in time?
*   Enroll in MSU again for fall semester

### Source Code
* Organize github repo
* Test the README in github
* Let it sit for two days
* Proofread again

