#image de base ubuntu
FROM ubuntu:latest

RUN mkdir /in
RUN mkdir /out

WORKDIR /app

COPY . /app/

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip3 install pandas pathlib numpy matplotlib plotly bids nibabel nilearn 
RUN pip3 install nireports
RUN pip3 install jinja2
RUN pip3 install --upgrade setuptools
RUN pip3 install nipype nitransforms


VOLUME ["/in", "/out"]

CMD ["python3", "/app/test.py", "/in"]
