# ==============================
# BASE IMAGE
# ==============================
FROM python:3.11-slim

# ==============================
# ENV SETTINGS
# ==============================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==============================
# WORK DIRECTORY
# ==============================
WORKDIR /app

# ==============================
# COPY PROJECT FILES
# ==============================
COPY . /app

# ==============================
# INSTALL DEPENDENCIES
# ==============================
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ==============================
# CREATE REQUIRED DIRECTORIES
# ==============================
RUN mkdir -p models/users
RUN mkdir -p data

# ==============================
# EXPOSE PORTS
# ==============================
EXPOSE 8000
EXPOSE 8501

# ==============================
# START SERVICES
# ==============================
CMD ["sh", "-c", "uvicorn app.app:app --host 0.0.0.0 --port 8000 & streamlit run app/frontend.py --server.port 8501 --server.address 0.0.0.0"]