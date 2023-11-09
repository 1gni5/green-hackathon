# Dockerfile to run ML jupyter notebook
FROM ubuntu:23.10

# Install python3 and pip3
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

# Install jupyter notebook
COPY requirements.txt /app/requirements.txt
RUN python3 -m venv /app/venv && /app/venv/bin/pip3 install --upgrade pip \
    && /app/venv/bin/pip3 install -r /app/requirements.txt

# Copy the notebook to the container
COPY . /app

# Expose the port
EXPOSE 8888

# Run the notebook
CMD ["/app/venv/bin/jupyter", "notebook", "--allow-root", "--ip=0.0.0.0"]