# Use Python 3.8 as the base image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the bot script into the image
COPY bot.py .

# Run the bot script
CMD ["python", "bot.py"]
