# Mushroom Classifier - AWS Lambda Container

FROM public.ecr.aws/lambda/python:3.12

# Copy project definition
COPY pyproject.toml uv.lock ${LAMBDA_TASK_ROOT}/

# Install uv
RUN pip install uv

# Install dependencies using uv sync
# This creates a .venv in the project root (/var/task/.venv) automatically
RUN uv sync --frozen

# Add virtual environment site-packages to PYTHONPATH
# This allows Lambda to find the installed dependencies
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}/.venv/lib/python3.12/site-packages:${LAMBDA_TASK_ROOT}"

# Copy function code
COPY cloud_deployment/lambda_function.py ${LAMBDA_TASK_ROOT}/lambda_function.py

# Copy models
COPY models/ ${LAMBDA_TASK_ROOT}/models/

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]
