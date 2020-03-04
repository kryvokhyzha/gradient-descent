FROM alpine:3.7

# Install Python and OpenJDK 8
RUN apk add --update \
    python3 \
    python3-dev \
    py-pip \
    build-base \
    && rm -rf /var/cache/apk/*

COPY . /home/root
WORKDIR /home/root

RUN pip3 install --upgrade pip==20.0.2
RUN pip3 install -r requirements.txt