FROM python:3.12-alpine

RUN adduser -u 65534 -D -s /sbin/nologin nobody

RUN pip install --no-cache-dir numpy

WORKDIR /workspace
USER 65534:65534

CMD ["python", "-c", "print('physsandbox runner container')"]
