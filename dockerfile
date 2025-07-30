# for smaller image footprint
FROM python:3.12-slim

# set the working directory of the app
WORKDIR /insurance-app

# installing the system dependencies for lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libomp-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# copy the app requirements
COPY requirements.txt .
#install the app requirements
RUN pip install --no-cache-dir -r requirements.txt

# copy all of the application code to the app folder
COPY app/ ./app/
# copy all the model files to the models folder
COPY models/ ./models/

# expose the default streamlit port
EXPOSE 8501

# run the app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]