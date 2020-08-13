# Predicting Art Auction Prices

**This project is a work in progess**

<!-- This project attempts to satisfy the requirements for Montana State University
STAT 575. This research attempts to answer whether an artwork's aesthetic
qualities and subject matter contribute meaningfully its auction price. If so,
this may hint that art consumers, at least those who participate in the art
auctions examined in this study, have similar subjective interpretations of
artistic beauty. If not, this may indicate that sale price is dominately
affected by hedonic characteristics such as period and artist, or physical but
non-subjective characteristics such size and medium.

The initial research for this project was conducted in late 2018 and early 2019,
but was put on hold for the remainder of 2019 before being restarted in late
March of 2020. For this reason there are a number of changes in the style and
engineering approach taken toward data acquisition and cleaning. These
inconsistencies will likely be evident if you attempt to reproduce this research
exactly. -->

## Setup

Code is written in python 3.7. Parts of this project also require the following:

* Docker

We highly recommend using a virtual environment. To install dependencies, from
within your virtual environment run:

```sh
pip install -Ur requirements.txt
```

## Data Acquisition

The data were scraped from Christies auction house website. This project uses
Google Cloud Platform infrastructure when applicable, so full replication of
this research will require familiarity with `gcloud` (the command line tools
for Google Cloud Platform).

* Create a python3 virtual environment and install `requirements.txt`.
* Create a GCP service account with storage read/write permissions via the cloud console. Download the key.
* Build a docker image based on the `Dockerfile`, tag it, and upload it to Cloud Repository

```bash
# Set up google registry
gcloud auth login
gcloud auth configure docker
```

```bash
GOOGLE_KEY=$(base64 -i [SERVICE-KEY-FILE].json)
docker build -t us.gcr.io/art-auction-prices/paap --build-arg GOOGLE_KEY=${GOOGLE_KEY} .
docker push us.gcr.io/art-auction-prices/paap
```

* Create a virtual machine

```bash
# List VMs
gcloud compute instances list
# Deploy VM
gcloud compute instances create-with-container paap-1 \
    --container-image=us.gcr.io/art-auction-prices/paap \
    --container-restart-policy "never" \
    --machine-type=n1-standard-1 \
    --scopes=storage-rw,logging-write
# SSH into VM
gcloud compute --project art-auction-prices ssh paap-1
# stop VM
gcloud compute --project art-auction-prices instances stop paap-1
# delete VM
gcloud compute instances delete paap-1
```

* Verify that the crawl container is running

```bash
gcloud compute ssh paap-1 --command "docker container ps -a"
```

### Images

#### On GCS

Follow the same process as above, but override the default container entrypoint
with:

```bash
gcloud compute instances create-with-container paap-1 \
    --container-image=us.gcr.io/art-auction-prices/paap \
    --container-restart-policy "never" \
    --machine-type=n1-standard-1 \
    --scopes=storage-rw,logging-write \
    --container-command="scrapy" \
    --container-arg="crawl" --container-arg="christiesImages"
```

Alternatively you can run it locally with:

```bash
docker run --entrypoint scrapy us.gcr.io/art-auction-prices/paap crawl christiesImages
```

To get a list of all the images in a given bucket and write that list to a file,
run:

```
gsutil ls gs://paap/christies/data/img/full/ >> ./data/img_in_gcs.txt
```

#### Locally

To simplify the process for local development, there are a number of scripts
for processing the images, which more or less represent a directed pipeline
for image acquisition and resizing.

Given the raw json files of data scraped from Christies' website, process them
into a CSV where each row represents a lot.

```sh
# python art/process_christies.py -i data/raw/*.json -o data/process_christies_output.csv
usage: process_christies.py [-h] -i INPUT_FILES [INPUT_FILES ...] -o OUTPUT_PATH

Clean json data, scraped from Christies into a format that can be used for predictive analytics

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILES [INPUT_FILES ...], --input-files INPUT_FILES [INPUT_FILES ...]
                        Input newline delimited json files to process.
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        CSV to save to
```

Given a CSV of raw piece data, scrapped from Christies website, return a two
column csv where each column is an artwork with lot_id and the corresponding
image_url:

```sh
# python scripts/create_image_csv.py -h
usage: create_image_csv.py [-h] input output

Create a two column CSV (<lot_id>,<image_url>) of image metadata from an input CSV of raw art data.

positional arguments:
  input       Input csv with raw art data
  output      Output path

optional arguments:
  -h, --help  show this help message and exit
```

From a two column csv of lot id and image url, download images to a local
directory.

```sh
# python scripts/download_images.py -h
usage: download_images.py [-h] input output_dir

Given an input csv with image urls, download images and save them to a location

positional arguments:
  input       Input csv. Two columns (<lot_id>,<image_url>), no header.
  output_dir  Directory to save images to.

optional arguments:
  -h, --help  show this help message and exit
```

In practice, after downloading the raw images they were uploaded to cloud
storage as a backup.

Given the raw CSV data and a CSV of image URLs to uuids, clean the data and join
to get UUID:

```sh
# python art/scripts/clean_data.py -h
usage: clean_data.py [-h] input_csv image_urls output_csv

Clean raw tabular data

positional arguments:
  input_csv   Path to input csv with raw scrapped data, header on first row
  image_urls  Path to a csv with image urls
  output_csv  Path to save output csv containing only cleaned data

optional arguments:
  -h, --help  show this help message and exit
```

Given a CSV with data, the following runs an interactive script which opens the
sale in a browser and asks whether the sale contains 2 dimensional works of art.
This is a coarse filtering mechanism that is used to exclude sales that are
comprised mostly of furniture, sculpture, ceramics, and books, which will
damage the ability of the model.

```sh
# python art/scripts/sale_is_2d.py --help
usage: sale_is_2d.py [-h] input_json output_json

Iterate over sales in a dataframe, and determine if they are exclusively 2d artwork

positional arguments:
  input_json   Input JSON file with sale_number/sale_url column
  output_json  Output JSON file to write to

optional arguments:
  -h, --help   show this help message and exit
```

We filtered images further based on keywords that indicated they might not be
two dimensional:

```sh
# python art/scripts/filter_artwork.py -h
usage: filter_artwork.py [-h] input_json is_2d_json output_json

Filter out artwork that is unsuitable for analysis according to a set of rules

positional arguments:
  input_json   Path to input json, the output from clean_data.py
  is_2d_json   Path to is_2d json, the output from is_2d.py
  output_json  Path to save output json containing only filtered data

optional arguments:
  -h, --help   show this help message and exit
```

Resize images to a common minimum dimension (i.e. the smallest of the two image
dimensions will have this pixel size):

```sh
# python scripts/resize_images.py -h
usage: resize_images.py [-h] [--image-size IMAGE_SIZE] [--delete] images output_dir

Resize images to a common minimum dimension, retaining aspect ratio

positional arguments:
  images                Images to process a newline separated file of image paths. A command like the following should get you started: `find data/img/christies/raw/ -type f -name '*.jpg' > data/img/christies/raw_images.txt`
  output_dir            Directory to save cropped images to

optional arguments:
  -h, --help            show this help message and exit
  --image-size IMAGE_SIZE
                        Images will be scaled to be this large in their minimum dimension, in pixels
  --delete              Delete the input photo after processing
```

We provide a script for randomly sampling the data and determining the
proportion of non-2d artwork.

```sh
# python -m art.scripts.sample_is_2d -h
usage: sample_is_2d.py [-h] [--input INPUT] [--output OUTPUT]

Randomly sample images from a dataset and determine if they are 2d or not

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    Path to input dataframe
  --output OUTPUT  Where to write the output
```


### Exchange Rates

To standardize prices, we calculate exchange rates, relative to a given date in
time:

```sh
# python art/scripts/exchange_rates.py -h
usage: exchange_rates.py [-h] [-o OUTPUT] [-d TARGET_DATE]

Get the dollar equivalent of currencies at a given date, and the inflation relative to today. Output is a csv with 5 columns: year, month, currency, dollar_equivalent, inflation. currency is the currency which we wish to convert to
dollars. dollar_equivalent is the price of that currency, in USD at that year and month. inflation is the multiple by which to multiply a dollar in that year/month to get it's equivalent worth at the target date.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
  -d TARGET_DATE, --target-date TARGET_DATE
```

Then we join those back into the cleaned dataset:

```sh
# python art/scripts/calculate_modern_price.py -h
usage: calculate_modern_price.py [-h] [--filtered_artwork FILTERED_ARTWORK] [--exchange_rates EXCHANGE_RATES] [--output OUTPUT]

Calculate prices in Jan 1, 2020 USD

optional arguments:
  -h, --help            show this help message and exit
  --filtered_artwork FILTERED_ARTWORK
                        Path to filtered artwork json, the output from filter_artwork
  --exchange_rates EXCHANGE_RATES
                        Path to the output of exchange rates
  --output OUTPUT       Output path to write the dataset to
```

Center crop images to a minimum dimension:

```sh
# python -m art.scripts.crop_images -h
usage: crop_images.py [-h] images output_dir

Crop images so they are square

positional arguments:
  images      Images to process a newline separated file of image paths. A
              command like the following should get you started: `find
              data/img/christies/raw/ -type f -name '*.jpg' >
              data/img/christies/raw_images.txt`
  output_dir  Directory to save cropped images to

optional arguments:
  -h, --help  show this help message and exit
```

## Running the model on GKE

From the [nn](nn) directory:

```sh
# Create a cluster
gcloud container clusters create paap-training-cluster \
    --num-nodes=1 \
    --zone=us-central1-f \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --machine-type="n1-highmem-8" \
    --scopes="gke-default,storage-rw" \
    --preemptible

# installing GPU nodes
# https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#ubuntu
# device drivers
# use `gcloud container get-server-config` to get the default image type
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# Resize the cluster
gcloud container clusters resize paap-training-cluster --num-nodes=0
```

Deploy a job to the cluster

Deploy an image to the cluster

```sh
kubectl apply -f ./job.yaml
watch --interval 10s "kubectl get jobs"
watch --interval 10s "kubectl get pods"
watch --interval 10s "kubectl describe pod <pod-id> | grep -A20 Events"
kubectl describe pod dcec-pod
```
