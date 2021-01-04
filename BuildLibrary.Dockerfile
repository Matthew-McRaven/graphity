# Create a base image with all of our requirements installed.
FROM python:3.8.5 as base
COPY ./requirements.txt /graphity/install/requirements.txt
RUN pip install -r /graphity/install/requirements.txt

# Copy over source files, build python package.
FROM base as output
COPY . /graphity/source
WORKDIR /graphity/source
RUN pip install -e .