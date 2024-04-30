# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the local directory contents into the container
COPY . /app


# Install necessary system dependencies for general operation
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libglib2.0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 to be accessible externally
EXPOSE 8000

# Run the application using uvicorn
CMD ["uvicorn", "joint_angles_realtime:app", "--host", "0.0.0.0", "--port", "8000"]
