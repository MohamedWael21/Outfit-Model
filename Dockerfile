# Use official Python image
FROM python:3.11-slim

# Install system dependencies for opencv and scikit-image
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
        && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Set environment variables for Flask
ENV FLASK_APP=main.py
ENV FLASK_ENV=production

# Start the app with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app", "--timeout", "120"] 