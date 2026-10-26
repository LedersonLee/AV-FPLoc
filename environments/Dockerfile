FROM python:3.10-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/avfploc
COPY environments/requirements-cu124.txt /tmp/requirements.txt
RUN pip install --no-cache-dir pip==25.2 && pip install --no-cache-dir torch==2.4.1+cu124 torchvision==0.19.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124 && pip install --no-cache-dir -r /tmp/requirements.txt
COPY . .
RUN pip install --no-deps .
ENV PYTHONUNBUFFERED=1
CMD ["avfploc", "--help"]
