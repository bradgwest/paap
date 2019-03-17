# Predicting Art Auction Prices

This project satisfies the requirements for Montana State University course STAT 575. The project seeks to predict art prices using a large dataset of historical and publicly available auction prices. The dataset includes a variety of predictor types from sale metadata to artwork characteristics.

## Setup

The data were scraped from an auction house website. This project used Google Cloud Platform infrastructure, including Compute Engine, Storage, and BigQuery services. To replicate the data acquisition, you will need to perform the following:

* Create a python3 virtual environment and install `requirements.txt`.
* Build a docker image based on the `Dockerfile`, tag it, and upload it to Cloud Repository

```bash
# Set up google registry
gcloud auth login
gcloud auth configure docker
docker build -t us.gcr.io/art-auction-prices/paap .
docker push us.gcr.io/art-auction-prices/paap
```

* Use the cloud console to create a virtual machine based on the recently created docker image
* Run the scraper to scrape data from cloud storage.

## Analysis

The data analysis is outlined in an Rmarkdown document, `analysis.Rmd`.

