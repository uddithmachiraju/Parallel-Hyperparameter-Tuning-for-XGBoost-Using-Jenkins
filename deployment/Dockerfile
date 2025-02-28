# Using lightweight python image 
FROM python:3.10-slim

# Set the working directory
WORKDIR /app 

# Copy necessary files
COPY deployment/app.py /app/
COPY models/best_xgb_model.pkl /app/models/
COPY config/requirements.txt /app/ 

# Install dependencies
RUN pip install -r requirements.txt 

# Expose the port 
EXPOSE 5000 

# Run the flask app 
CMD ["python", "app.py"] 
