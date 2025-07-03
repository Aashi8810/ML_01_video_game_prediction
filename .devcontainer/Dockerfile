FROM python:3.10

# Set working directory
WORKDIR /workspace

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional tools
RUN pip install --no-cache-dir cookiecutter jupyterlab ipykernel
