# Use a specific version of Python
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# # Default command to run main.py first, then init_db.py and app.py
# CMD ["sh", "-c", "python main.py && python init_db.py && python app/app.py"]

# Default command to run main.py first, then init_db.py and app.py
CMD ["sh", "-c", "python main.py && python app/app.py"]
