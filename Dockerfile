FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY ai_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start FastAPI server using uvicorn
CMD ["python", "-m", "uvicorn", "ai_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
