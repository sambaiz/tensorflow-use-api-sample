FROM gcr.io/tensorflow/tensorflow

ADD requirements.txt /tmp
ADD model.ckpt /
ADD training/mnist.py /
ADD app.py /

RUN pip install -q -r /tmp/requirements.txt

EXPOSE 3000

CMD ["python", "/app.py"]
