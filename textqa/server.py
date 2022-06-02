from aiohttp import web
from deeppavlov import build_model
import os
import json


model = build_model(os.environ.get('CONFIG', ''), download=True)


def array_to_string(array):
    return ",".join([str(x) for x in array])


async def call_test(request):
        content = "get ok"
        return web.Response(text=content, content_type="text/html")


async def call_inference(request):
        response = ''
        request_str = json.loads(str(await request.text()))
        request = json.loads(request_str)	
        if not 'texts' in request.keys():
                response += 'Parameter not found in request: texts\n'
        if not 'questions' in request.keys():
                response += 'Parameter not found in request: questions\n'
        if response == '':
                response = model(request['texts'], request['questions'])
        response = json.dumps(response)
        return web.Response(text=str(response),content_type="text/html")


def main():
        app = web.Application(client_max_size=1024**3)
        app.router.add_route('GET', '/test', call_test)
        app.router.add_route('POST', '/inference', call_inference)
        web.run_app(app, port=os.environ.get('PORT', ''))


if __name__ == "__main__":
        main()






# Sentiment analysis
# https://demo.deeppavlov.ai/#/ru/sentiment

from deeppavlov import build_model, configs
import pandas as pd
import datetime
import time
#import pymssql
import os

BATCH_SIZE = 1000

def main():

        model = build_model(configs.classifiers.rusentiment_bert, download=True)

        line = 0
        while True:

                query = """
                        select top """+str(BATCH_SIZE)+"""
                        id,     text, sentiment
                        from transcribations
                        where sentiment is NULL and text!=''
                        order by transcribation_date, start
                        """

                df = pd.read_sql(query, conn)

                if len(df) > 0:
                        print(datetime.datetime.now(), 'solving '+str(len(df))+' records')
                        df['sentiment'] = model(df.text)
                        print(datetime.datetime.now(), 'updating')
                        update_record(conn, df)

                else:
                        print(datetime.datetime.now(), 'No data. waiting..')
                        time.sleep(10)

                print(datetime.datetime.now(), 'next job..')

if __name__ == "__main__":
	main()
