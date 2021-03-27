FROM tensorflow/tensorflow:2.3.0-gpu

WORKDIR /app

COPY . .

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install -r requirements.txt

CMD ["python", "./app.py"]