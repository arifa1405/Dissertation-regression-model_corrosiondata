FROM python:3.10-slim

WORKDIR /app

COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deployment/ ./deployment/

EXPOSE 5000
CMD ["python", "deployment/app.py"]
