import subprocess
import os

# myenv'i aktive etmeden doğrudan python-embed ile başlat
streamlit_cmd = r".\python-embed\python.exe"
app_path = "app.py"

print("🚀 Akıllı CV Analiz Sistemi Başlatılıyor...")
subprocess.run([streamlit_cmd, "-m", "streamlit", "run", app_path])
