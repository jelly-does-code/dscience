# Use a lightweight Python image
FROM python:3.9-slim

ENV MODEL_PATH="XGBClassifier_best.json"

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY docker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model
COPY docker/main.py ./main.py
COPY ../codebase/models/XGBClassifier_best.json ./XGBClassifier_best.json

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
