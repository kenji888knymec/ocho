# gunicorn.conf.py

workers = 1
threads = 1

# これが「worker timeout」です（今回の主題）
timeout = 1200

# graceful shutdown
graceful_timeout = 30

# ログ（必要なら）
loglevel = "info"

# Cloud Run では preload は基本NG（メモリ急増/起動不安定化）
preload_app = False
