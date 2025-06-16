# Use AWS Lambda Python 3.12 base image
FROM public.ecr.aws/lambda/python:3.12

WORKDIR /var/task

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code and environment variables
COPY main.py .
COPY .env .

# Set the Lambda handler
CMD ["main.lambda_handler"]
