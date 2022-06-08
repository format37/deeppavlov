# Natural language processing services
This is a list of several ready to use NLP services, wich i used in practice and built as distinct docker containers, preferrly required Nvidia GPU adapter.
### Installation
```
git clone https://github.com/format37/nlp.git
cd nlp
mv docker-compose.yml.default docker-compose.yml
```
Open and edit docker-compose.yml  
To enable the selected service, set according service replicas: 1  
Define, wich nvidia gpu adapter used for service, starting from 0. device_ids: ['0']
## Deeppavlov paraphrase
This GPU service receives a two lists of phrases.
And returns a list[n], wich defines, is list_a[n] and list_b[n] are paraphrase or not.  
Usage example: [examples/paraphrase.ipynb](https://github.com/format37/nlp/blob/main/examples/paraphrase.ipynb)
### Deeppavlov sentiment
### Deeppavlov textqa
### Summarus
