FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app/code
EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "code/app.py", "--server.address=0.0.0.0"]
