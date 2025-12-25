web: gunicorn -b :$PORT --workers 1 --threads 1 --timeout 1200 --graceful-timeout 30 main:app
train: python3 -u train_ai_model.py

