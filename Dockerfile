# 3db Unified Database Ecosystem - Multi-stage Dockerfile
# Optimized for production deployment with development support

# =====================================================================================
# BASE STAGE - Common dependencies
# =====================================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# =====================================================================================
# DEVELOPMENT STAGE
# =====================================================================================
FROM base as development

# Install development dependencies
COPY requirements-dev.txt /app/
RUN pip install -r requirements-dev.txt

# Install additional development tools
RUN pip install \
    jupyter \
    ipython \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pre-commit

# Copy source code
COPY --chown=app:app . /app/

# Switch to app user
USER app

# Expose development ports
EXPOSE 8000 8001 8888

# Development command
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =====================================================================================
# TESTING STAGE
# =====================================================================================
FROM base as testing

# Install testing dependencies
COPY requirements-dev.txt /app/
RUN pip install -r requirements-dev.txt

# Copy source code
COPY --chown=app:app . /app/

# Switch to app user
USER app

# Run tests
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=src", "--cov-report=html", "--cov-report=term"]

# =====================================================================================
# PRODUCTION STAGE
# =====================================================================================
FROM base as production

# Install production dependencies only
RUN pip install gunicorn

# Copy source code
COPY --chown=app:app src/ /app/src/
COPY --chown=app:app api/ /app/api/
COPY --chown=app:app config/ /app/config/
COPY --chown=app:app scripts/ /app/scripts/

# Copy configuration files
COPY --chown=app:app *.py /app/
COPY --chown=app:app .env.example /app/.env.example

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/tmp

# Switch to app user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose application port
EXPOSE 8000

# Production command with Gunicorn
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", \
     "--keep-alive", "2", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "api.main:app"]

# =====================================================================================
# MONITORING STAGE
# =====================================================================================
FROM base as monitoring

# Install monitoring dependencies
RUN pip install \
    prometheus-client \
    psutil \
    py-spy

# Copy source code
COPY --chown=app:app . /app/

# Switch to app user
USER app

# Expose monitoring port
EXPOSE 8001

# Monitoring command
CMD ["python", "scripts/monitoring.py"]

# =====================================================================================
# BUILD ARGUMENTS AND LABELS
# =====================================================================================

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Labels
LABEL maintainer="3db Development Team" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="3db Unified Database Ecosystem" \
      org.label-schema.description="Intelligent database system combining PostgreSQL, pgvector, and Apache AGE" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/your-org/3db" \
      org.label-schema.schema-version="1.0"
