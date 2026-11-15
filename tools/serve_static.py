"""Статичний сервер для Project/web без кешування HTML/JS/CSS/JSON (браузер завжди бере свіжу версію).
PNG-шари кешуються на годину.

    python tools/serve_static.py --port 8080 --bind 0.0.0.0 --dir web
"""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        p = self.path.split('?', 1)[0]
        if p.endswith(('.png', '.jpg', '.webp')):
            # перевіряти свіжість щоразу (відповідь 304 дешева): після нового експорту шарів
            # браузер не покаже стару картинку з кешу
            self.send_header('Cache-Control', 'no-cache')
        else:
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--bind', default='0.0.0.0')
    ap.add_argument('--dir', required=True)
    a = ap.parse_args()
    ThreadingHTTPServer((a.bind, a.port), functools.partial(Handler, directory=a.dir)).serve_forever()


if __name__ == '__main__':
    main()
