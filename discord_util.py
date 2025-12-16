import requests
from config import DISCORD_WEBHOOK_URL

http = requests.Session()
http.headers.update({"User-Agent": "spidey-bot/v3.4.3"})

def send_discord_message(text: str):
    if (not DISCORD_WEBHOOK_URL) or ("ここに" in DISCORD_WEBHOOK_URL):
        print("[DBG] discord webhook url empty or placeholder")
        return

    chunks = [text[i:i+1900] for i in range(0, len(text), 1900)] or [text]
    for chunk in chunks:
        try:
            r = http.post(DISCORD_WEBHOOK_URL, json={"content": chunk}, timeout=10)
            print(f"[DBG] discord status={r.status_code}")
            if r.status_code >= 300:
                print(f"[DBG] discord body={r.text[:200]}")
        except Exception as e:
            print(f"[ERR] discord webhook post: {e}")
