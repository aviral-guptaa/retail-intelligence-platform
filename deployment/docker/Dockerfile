FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# Uncomment the optional section in requirements.txt if the image needs YOLO.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "main.py", "--mode", "demo"]