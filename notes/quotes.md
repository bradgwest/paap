# Quotes From Papers

## Castellano, 2020. Deep Convolutional Embedding For Digitized Painting Clustering

>   On the other hand, the application of traditional clustering and feature 
    reduction techniques to the highlydimensional pixel space can be ineffective.

>   The ability to recognize meaningful patterns in visual artworks inherently
    falls within the domain of human perception[1]. Recognizing stylistic and
    semantic attributes of a painting, in fact, originates from the
    composition of the colour,texture and shape features visually perceived
    by the human eye. These attributes, which typically concern the distribution
    of colours, the spatial complexity of the painted scene, etc., together
    are responsible for the overall “visual appearance”of the artwork.

>   Having a model capable of clustering artworks in accordance with their
    visual appearance, without the need of collecting labels andmetadata, can
    be useful for many applications.

>   It can be used to support art experts in findings trends and influences
    among painting schools, i.e. in performing historical knowledge discovery.
    Analogously, it can be used to discoverdifferent periods in the production of a same artist. The model may discover which artworks influenced mostly thework of current artists. It may support interactive navigation on online art galleries by finding visually linked artworks,i.e. visual link retrieval. It can help curators in better organizing permanent or temporary expositions in accordancewith their visual similarities rather than historical motivations

>  Conversely, several successful applications in a number of Computer Vision tasks (e.g., [15,16,17,18]) have shownthat representation learning is an effective alternative to feature engineering to extract meaningful patterns from complexraw data

>   The aggregation of all currently available art collections would result in asignificantly smaller number of images compared to ImageNet. Instead, a model built on these data often provides asufficiently general knowledge of the “visual world”, which can be transferred to specific visual domains profitably

>   The main issue to be addressed in this kind of research is the so-calledcross-depictionproblem, that is the problem of recognizing visual objects regardless of whether they are photographed, painted, drawn,etc. The variance across photos and artworks is greater than either domains if considered alone, thus classifiers usuallytrained on traditional photographic images may find difficulties when used on painting images, due to the domain shift.

>   On the other hand, applying well-known dimensionality reduction techniques, such as PCA [34],either to the original space or to a manually engineered feature space, can ignore possible nonlinear transformationsfrom the original input to the latent space, thus decreasing clustering performance.

>   In recent years, a deep clustering paradigm has emerged which takes advantage of the capability of deep neural networksof finding complex nonlinear relationships among data for clustering purposes [11,35,36].  The idea is to jointlyoptimize the task of mapping the input data to a lower dimensional space and the task of finding a set of centroids inthis latent feature space

>   We assume an inputconsisting of128×128three-channel scaled images, normalized in the range[0,1].  This input is then propagatedthrough a stack of convolutional layers which learn to extract hierarchical visual features. The first convolutional layerhas32filters, with kernel size5×5. The second convolutional layer has 64 filters, with kernel size5×5. The thirdconvolutional layer has128filters, with kernel size3×3. The number of filters in the last two layers is higher mainlybecause the number of low level features (i.e., circles, edges, lines, etc.) is typically low, but the number of ways tocombine them to obtain higher level features can be high

>   Experiments were run on an Intel Core i5 equipped with the NVIDIA GeForce MX110, with dedicated memory of2GB. As deep learning framework, we used TensorFlow 2.0 and the Keras API [39]
