FROM python:3.11-slim
WORKDIR /app
# Now this will work because the file is in the same folder as Dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]