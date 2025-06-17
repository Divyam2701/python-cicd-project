# Use AWS Lambda Python 3.12 base image
FROM public.ecr.aws/lambda/python:3.12

# Set working directory
WORKDIR /var/task

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code, env file, and index.html
COPY main.py .
COPY .env .
COPY index.html .


# Set the handler (filename.function_name)
CMD ["main.lambda_handler"]
