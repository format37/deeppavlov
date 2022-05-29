# Sentiment analysis
# https://demo.deeppavlov.ai/#/ru/sentiment

from deeppavlov import build_model, configs
import pandas as pd
import datetime
import time
import pymssql
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
