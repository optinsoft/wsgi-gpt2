def application(environ, start_response):
    with open('html/test.html') as f:
        response_body = f.read().encode('utf-8')

    status = '200 OK'
    response_headers = [
        ('Content-Type', 'text/html'),
        ('Content-Length', str(len(response_body))),
    ]

    start_response(status, response_headers)
    yield response_body
