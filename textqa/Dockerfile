FROM deeppavlov/base-gpu:latest
WORKDIR /service
RUN python3 -m deeppavlov install squad_bert
RUN python3 -m deeppavlov install squad_ru
#RUN python3 -m deeppavlov install config_file
#RUN pip3 install med2image
RUN python3 -m pip install --upgrade pip
COPY requirements.txt /service
RUN pip3 install -r requirements.txt --no-cache-dir
#RUN apt-get update
#RUN apt-get install nano
COPY server.py /service
#RUN wget raw.githubusercontent.com/deepmipt/DeepPavlov/master/deeppavlov/configs/classifiers/rusentiment_bert.json
ENTRYPOINT ["python3"]
CMD ["server.py"]