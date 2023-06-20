# WSGI service for gpt-2

## Requirements

```text
python 3.7.16
tensorflow 1.15.0
protobuf 3.20.3
fire 0.5.0
regex 2023.6.3
requests 2.21.0
tqdm 4.31.1
```

## Usage

Download model `124M`:

```bash
python .\download_model.py 124M
```

Run wsgi gpt-2 api on port 8000:

```bash
python .\gpt2_server.py
```

Run test http server on port 9000:

```bash
python .\test_httpd.py
```

Use this link to access test http server: [http://localhost:9000/](http://localhost:9000/)
