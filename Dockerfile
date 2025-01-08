FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Update the package list
RUN apt-get update -y

# Install git
RUN apt-get install -y git

# Install r-base and tzdata
RUN apt-get install -y r-base tzdata

ADD requirements.txt .
RUN pip install -r requirements.txt


CMD
