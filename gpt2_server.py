from wsgiref.simple_server import make_server

from apps.gpt2_app import GPT2Application
from middleware.cors_middleware import CORSMiddleware

serve_port = 8000

print(f"Serving wsgi-gpt2 api on port {serve_port}...")
app = CORSMiddleware(GPT2Application(), whitelist=['http://localhost:9000'])
httpd = make_server('localhost', serve_port, app)
httpd.serve_forever()
