# Natural language processing services
This is a list of several ready to use NLP services, wich i used in practice and built as distinct docker containers, preferrly required Nvidia GPU adapter.
### Installation
```
git clone https://github.com/format37/nlp.git
cd nlp
mv docker-compose.yml.default docker-compose.yml
```
Open and edit docker-compose.yml  
To enable the selected service, set according service replicas parameter:
```
replicas: 1  
```
Define, wich nvidia gpu adapter used for service, starting from 0:
```
device_ids: ['0']
```
Define the model, corresponding to language you need, if available in docker-compose.yml config.  
To build server, run:
```
sudo docker-compose up --build -d
```
And test the service, with one of examples below
### Deeppavlov paraphrase
Docker service, receives two lists of phrases.  
Returns a list[n] of int values: 0 or 1, wich defines, is list_a[n] and list_b[n] are paraphrase or not.  
Example: [paraphrase.ipynb](https://github.com/format37/nlp/blob/main/examples/paraphrase.ipynb)
### Deeppavlov sentiment
Docker service, receives one list of phrases.  
Returns a list[n] of sentiment categories ['positive', 'neutral', 'negative', 'speech'], for each list[n] phrase.  
Example: [sentiment.ipynb](https://github.com/format37/nlp/blob/main/examples/sentiment.ipynb)
### Deeppavlov textqa
Docker service, receives two lists:
* texts
* questions  
  
Returns two lists:
* questions
* answers
Example: [textqa.ipynb](https://github.com/format37/nlp/blob/main/examples/textqa.ipynb)
### Summarus
Docker service, receives one string of Russian text.  
Returns one summarized string of text.  
Example: [summarus.ipynb](https://github.com/format37/nlp/blob/main/examples/summarus.ipynb)
