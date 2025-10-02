# Базовый образ с PyTorch и CUDA (можно без GPU, если не надо)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Установим системные зависимости
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Сначала скопируем requirements.txt (собери его заранее)
COPY requirements.txt .

# Ставим Python-зависимости
RUN pip install -r requirements.txt

# Копируем все ноутбуки, код и модели
COPY *.ipynb ./
COPY microservice.py ./
COPY *.pth ./

# Открываем порт
EXPOSE 8000

# Запускаем сервис
CMD ["uvicorn", "microservice:app", "--host", "0.0.0.0", "--port", "8000"]
