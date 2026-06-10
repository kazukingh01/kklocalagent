FROM python:3.11-slim-bookworm
WORKDIR /app
RUN pip install --no-cache-dir 'aiohttp>=3.9,<4'
COPY harness.py /app/harness.py
CMD ["sleep", "infinity"]
