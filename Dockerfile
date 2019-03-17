FROM python:3.7-stretch

# setup env
RUN mkdir -p /paap/art
RUN mkdir -p /data/
WORKDIR /paap

# Add src files
ADD art/* ./art/
ADD scrapy.cfg .
ADD requirements.txt .

RUN pip install -r requirements.txt

ENTRYPOINT ["scrapy", "crawl", "christies", "-o", "/data/christies.json"]
