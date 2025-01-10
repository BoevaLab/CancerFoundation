FROM nvcr.io/nvidia/pytorch:24.06-py3

ENV DEBIAN_FRONTEND=noninteractive

# Update the package list
RUN apt-get update -y

# Install git
RUN apt-get install -y git

# Install r-base and tzdata
RUN apt-get install -y r-base tzdata
RUN python -m pip install --upgrade pip
ADD requirements.txt .
RUN pip install -r requirements.txt