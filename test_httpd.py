import logging
import time

from wsgiref.simple_server import make_server

from apps.test_app import application

log = logging.getLogger()

serve_port = 9000

print(f"Serving test httpd on port {serve_port}...")
httpd = make_server('localhost', serve_port, application)
httpd.serve_forever()
