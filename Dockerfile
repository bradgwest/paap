FROM python:3.7-stretch AS python-paap

# GOOGLE_KEY must be passed in as ENV variable, base64 encoded
# Set default app credentials for container
ARG GOOGLE_KEY
ENV GOOGLE_APPLICATION_CREDENTIALS=/.google/google-key.json

# setup env
RUN mkdir /.google
RUN mkdir -p /paap/art/log
WORKDIR /paap

# Add src files
ADD art ./art
ADD scrapy.cfg .
ADD requirements.txt .

RUN pip install -r ./requirements.txt

# Set google key
RUN echo ${GOOGLE_KEY} | base64 --decode > /.google/google-key.json

# Set py path so scrapy can be called
ENV PYTHONPATH "${PYTHONPATH}:/paap/art"

WORKDIR /paap/art

ENTRYPOINT ["scrapy", "crawl", "christies"]
