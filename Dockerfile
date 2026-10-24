FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY bot.py odoo_tools.py tts_factory.py compare_tts.py ./

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import socket; socket.create_connection(('127.0.0.1', 7860), 3)"

# Pipecat runner: --transport twilio | webrtc. Override the command to run compare_tts.py.
CMD ["python", "bot.py", "--host", "0.0.0.0", "--transport", "webrtc"]
