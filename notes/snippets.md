# Snippets

Notes about the process that might be used in the paper.

## Keywords

- highly dimensional pixel space
- latent space
- visual link retrieval
- visual arts
- meaningful patterns
- artistic style categorization
- object detection
- art movement
- art genre
- feature extraction
- representation learning
- feature extraction vs representation learning
- cross-depiction problem - the problem of recognizing visual object regardless of whether they are photographed, painted, drawn, etc
- discriminative model
- generative model
- art periods
- local receptive field
- feature map - map from theinput layer to the hidden layer
- shared weights - the weights in the feature map which are hidden
- shared bias - the single shared bias that the feature map has for each local receptive field
- kernal

## Data Prep

*    During the data prep phase, a number of inconsistencies (read bug) in the javascript
     and static pages of the Christies website made cleaning an arduous process.
     For instance, the location of artist and description in the server side
     data structure changed. We attempted to acommodate for this with some
     clever regex processing and time consuming manual validation.

*    In mid 2017 Christies changed their front end architecture from serving
     static html pages to serving javascript rich pages, essentially a single
     page application. this added additional processing burdern as we needed
     to develop multiple crawling processors to handle the differences.

*   Where possible we attempted to clean data with automatic tooling, but used
    manually efforts in a number of ways. Rather than use NLP on the description
    or title of the artwork, we decided to identify (and filter out) 3 dimensional
    pieces manually, at the sale level. That is to say we browsed each of 2000+
    sales and determined if the artwork was exclusively 2 dimensional, discarding
    the sale if not. This process was not foolproof (looking through 400k images)
    is a time consuming task, so there are likely some images in the dataset
    which are pictures of 3 dimensional pieces of work.

    This is an interesting thing two dwell on for a second. The human eye is so
    good at recognizing details that we can tell what is a picture of a three
    dimensional object vs what it a picture of an image of a three dimensional
    object. Of course there may be pictures of pictures of 3d objects, in which
    case we mistakenly excluded that piece of artwork, but, nonetheless this
    exercise is a testament to how crude computer vision methods are.

*   One source of potential noise in the model is the inclusion of the frame
    in the image. Christies will occasionally include the frame in the image
    of the artwork. We consider this noise, as it's not exclusively the artwork,
    but it could be argued that this is the way Christies presents the data,
    and therefore is not data.

*   We stripped text of whitespace.

## Motivation

*   In recent years there have been a number of attempts to develop AIs that are
    capable of making artwork. Results have been ___. This work has the potential
    to aid in those efforts. If we're able to identify features of artwork that
    are more highly priced, then we can tune AI generative models to more highly
    weight those features which were highly priced.

*   In the past (time period), neural networks have become the gold standard for
    computer vision tasks, including clustering. We chose this method simply
    because it's better than other methods.

## Model

*   Clustering images is a distinctly different effort than traditional computer
    vision tasks. Many computer vision problems involve object recognition, or
    object detection. While clustering images involves object recognition and
    detection, we care primarily about the stylistic depiction of those
    objects less so about their actual types.
