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
        request_str = json.loads(str(await request.text()))
        data = json.loads(request_str)
        result = model(data['text_a'], data['text_b'])
        answer = array_to_string(result)
        return web.Response(text=answer, content_type="text/html")


def main():
        app = web.Application(client_max_size=1024**3)
        app.router.add_route('GET', '/test', call_test)
        app.router.add_route('POST', '/inference', call_inference)
        web.run_app(app, port=os.environ.get('PORT', ''))


if __name__ == "__main__":
        main()
