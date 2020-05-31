FROM python:3.7-slim

EXPOSE $PORT

COPY ./src /home/root
COPY requirements.txt /home/root
WORKDIR /home/root

RUN pip install --upgrade pip==20.1.1
RUN pip install -r requirements.txt

RUN echo $PORT

CMD python -m streamlit.cli run --server.enableCORS false --server.port $PORT app.py
