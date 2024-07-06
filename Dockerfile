FROM python:3.11-slim

WORKDIR /code

# Install pip requirements
COPY requirements.txt /code
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /code/requirements.txt && \
    apt-get update && apt-get install -y --no-install-recommends libgomp1

# Copy local code to the container image.
COPY . /code

# Make workspace directory
RUN mkdir /code/workspace

# Move the /notebooks directory to the workspace
RUN mv /code/notebooks /code/workspace

# Get inside the workspace
WORKDIR /code/workspace

# Expose the port Jupyter Notebook uses
EXPOSE 8888

# Start bash by default
CMD ["/bin/bash"]