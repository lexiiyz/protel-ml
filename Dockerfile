# Gunakan Python Image yang ringan
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# 1. Install System Dependencies (Wajib buat OpenCV & EasyOCR)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy Requirements & Install Python Libs
COPY requirements.txt .
# Install torch versi CPU saja biar image kecil (kecuali servermu punya GPU NVIDIA & mau pake nvidia-docker)
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Semua Source Code
COPY . .

# 4. Expose Port (Sesuai api.py)
EXPOSE 8000

# 5. Jalankan API
CMD ["python", "api.py"]