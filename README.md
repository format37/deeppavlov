# Natural language processing services

### Installation
```
git clone https://github.com/format37/nlp.git
cd nlp
mv docker-compose.yml.default docker-compose.yml
```
Open and edit docker-compose.yml  
To enable the selected service, set according service replicas: 1
Define, wich nvidia gpu adapter used for service, starting from 0. device_ids: ['0']
## Deeppavlov paraphrase detection
This service receives a two lists of phrases,  
and returns a list[n], wich defines, is list_a[n] and list_b[n] are paraphrase or not.
