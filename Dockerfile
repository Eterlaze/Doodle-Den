FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖（opencv 等需要）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制你的代码
COPY . .

# 暴露端口（云托管默认使用 80 或你指定的端口）
EXPOSE 80

# 启动命令
CMD ["python", "app.py"]