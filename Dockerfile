FROM python:3.8-slim

COPY ./src /home/root
COPY requirements.txt /home/root
WORKDIR /home/root

RUN pip install --upgrade pip==20.1.1
RUN pip install -r requirements.txt

CMD ["python", "-m", "streamlit.cli", "run", "app.py"]
