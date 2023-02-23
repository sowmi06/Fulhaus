FROM python:3.8

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

RUN apt-get update && apt-get install python3-opencv  -y

COPY . .


RUN python3 model/classification_model.py

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]