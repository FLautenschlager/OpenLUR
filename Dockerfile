FROM python:3

RUN apt update && apt install -y python3-wheel swig python3-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /code
WORKDIR /code

CMD python3 two_cities_experiment.py