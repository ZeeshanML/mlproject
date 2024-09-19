# Use the official Python 3.10.0 image from the Docker Hub
FROM python:3.10.0-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN apt update -y && apt install awscli -y

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Run app.py when the container launches
CMD ["python3", "app.py"]