# Use official Python image
FROM python:3.11-slim

# Avoid writing .pyc files & get unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps
COPY requirements-docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy project
COPY . /app

EXPOSE 8000

# ✅ Launch HW3 API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

