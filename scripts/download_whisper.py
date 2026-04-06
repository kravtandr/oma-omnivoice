"""Скачать Whisper large-v3-turbo в ../whisper-model/ для офлайн-использования."""

from huggingface_hub import snapshot_download
import os

model_id = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3-turbo")
output_dir = os.path.join(os.path.dirname(__file__), "..", "whisper-model")

print(f"Скачиваю {model_id} → {os.path.abspath(output_dir)}")
snapshot_download(repo_id=model_id, local_dir=output_dir)
print("Готово.")
