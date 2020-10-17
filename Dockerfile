# Create a base image with all of our requirements installed.
FROM python:3.8.5 as base
COPY ./requirements.txt /graphity/install/requirements.txt
RUN pip install -r /graphity/install/requirements.txt

# Copy over source files, build python package.
FROM base as build
COPY . /graphity/source
WORKDIR /graphity/source
RUN pip wheel . -w /graphity/install

# Output image only contains the built wheel and installed requirements.
FROM build as output
COPY --from=base /graphity/install/ /graphity/install
RUN pip install $(find /graphity/install -type f -iname "*.whl")