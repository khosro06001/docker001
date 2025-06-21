############# FROM python:3.9-slim-buster
### FROM arm64v8/python:3.x-slim-bookworm
### FROM arm64v8/python:3.14.0b2-slim-bookworm
### FROM arm64v8/python:3.14-rc-slim-bookworm
FROM python:3.14-rc-slim-bookworm

WORKDIR /app

################COPY test001.py .
COPY test002.py .

### RUN pip install requirements.txt
### EXPOSE 80  ### expose port 80
### CMD ["python","test001.py"]

CMD ["python","test002.py"]


