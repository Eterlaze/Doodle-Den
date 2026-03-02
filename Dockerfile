# 使用 Debian 11 (bullseye) 的 slim 镜像，更稳定
FROM python:3.9-slim-bullseye

WORKDIR /app

# 安装系统依赖（OpenCV 需要）
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口（云托管默认使用 80）
EXPOSE 80

# 启动命令
CMD ["python", "app.py"]