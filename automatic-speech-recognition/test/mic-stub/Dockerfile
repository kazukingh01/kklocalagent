FROM python:3.12-slim
RUN pip install --no-cache-dir 'websockets>=12,<14'
WORKDIR /app
COPY stub.py .
EXPOSE 7010
ENV HOST=0.0.0.0 PORT=7010 SAMPLE_PATH=/samples/jfk.wav LOOP_DELAY_MS=2000
CMD ["python", "-u", "stub.py"]
