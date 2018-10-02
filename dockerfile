FROM python:3.5.6-slim

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
COPY . /app

EXPOSE 8000

CMD ["gunicorn", "main:api", "--bind", "0.0.0.0"]
