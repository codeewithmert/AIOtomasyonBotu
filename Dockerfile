# AI Automation Bot - Dockerfile
# Multi-stage build for optimized production image

# Base stage
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p logs data models reports backups

# Set development environment
ENV ENVIRONMENT=development
ENV PYTHONPATH=/app

# Development command
CMD ["python", "-m", "src.main", "--daemon"]

# Production stage
FROM base as production

# Install production dependencies only
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p logs data models reports backups

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set production environment
ENV ENVIRONMENT=production
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Production command
CMD ["python", "-m", "src.main", "--daemon"]

# Testing stage
FROM development as testing

# Install testing dependencies
RUN pip install --no-cache-dir \
    pytest-asyncio \
    pytest-mock \
    factory-boy \
    coverage

# Copy test configuration
COPY pytest.ini .
COPY .coveragerc .

# Test command
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=html"]

# Documentation stage
FROM base as documentation

# Install documentation dependencies
RUN pip install --no-cache-dir \
    sphinx \
    sphinx-rtd-theme \
    sphinx-autodoc-typehints

# Copy documentation files
COPY docs/ ./docs/
COPY README.md .

# Build documentation
RUN sphinx-build -b html docs/ docs/_build/html

# Serve documentation
EXPOSE 8080
CMD ["python", "-m", "http.server", "8080", "--directory", "docs/_build/html"] 