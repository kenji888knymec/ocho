web: gunicorn -b :$PORT --workers 1 --threads 1 --timeout 1200 --graceful-timeout 30 main:app
train: /bin/sh -c "echo RUN_TRAIN; python -u train_ai_model.py; echo TRAIN_EXIT_CODE=$?"

