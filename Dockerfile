# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system deps (optional: good for building some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Expose all predictor ports
EXPOSE 3075 3076 3077

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3075", "--reload"]
