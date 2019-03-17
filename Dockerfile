FROM python:3.7-stretch AS python-paap

# setup env
RUN mkdir -p /paap/art
RUN mkdir -p /paap/log

WORKDIR /paap

# Add src files
ADD art ./art
ADD scrapy.cfg .
ADD requirements.txt .

RUN pip install -r ./requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/paap/art"

WORKDIR /paap/art

ENTRYPOINT ["scrapy", "crawl", "christies", "&>", "/paap/log/scrapy.log"]
