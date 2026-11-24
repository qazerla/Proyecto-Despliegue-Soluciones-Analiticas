FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RADAR_DATA=/app/data \
    API_URL=http://api:8000

COPY radar_dash/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY radar_dash /app

EXPOSE 8051

CMD ["waitress-serve", "--host=0.0.0.0", "--port=8051", "app:server"]