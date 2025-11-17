# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for torch, torchvision, spacy, numpy, pandas etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libxml2 \
    libxslt1.1 \
    libssl-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Install PyTorch CPU (Correct versions)
# -------------------------------
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cpu

# -------------------------------
# Install spaCy separately (avoids failures)
# -------------------------------
RUN pip install --no-cache-dir spacy==3.7.4

# -------------------------------
# Copy requirement file
# -------------------------------
COPY requirements.txt .

# Install other requirements (longer timeout)
RUN pip install --default-timeout=200 --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose application port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
