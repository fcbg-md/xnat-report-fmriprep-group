#image de base ubuntu
FROM ubuntu:latest

RUN mkdir /in
RUN mkdir /out

WORKDIR /app

COPY . /app/

RUN pip install pandas pathlib numpy matplotlib plotly bids nibabel nilearn 
RUN pip install nireports
RUN pip install jinja2
RUN pip install --upgrade setuptools
RUN pip install nipype nitransforms

VOLUME [ "/data_input", "/data_output" ]

ENTRYPOINT [python /app/test.py "$1"]
