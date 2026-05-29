# Streamlit Service Setup

This guide outlines how to set up a systemd service to run a Streamlit app on your machine.

## 1. Create the systemd Service File

Open the systemd service file for editing:

```bash
sudo nano /etc/systemd/system/streamlit.service

[Unit]
Description=Streamlit App
After=network.target

[Service]
User=pixii
WorkingDirectory=/home/AI/iDOC
ExecStart=/home/pixii/miniconda3/envs/pixii_ai/bin/streamlit run app.py
Restart=always
RestartSec=3
Environment="PATH=/home/pixii/miniconda3/envs/pixii_ai"

[Install]
WantedBy=multi-user.target

```
## 2. Run the following commands
```bash
sudo systemctl daemon-reload

sudo systemctl enable streamlit.service

sudo systemctl start streamlit.service

```

### 3. Debugging
```bash
sudo ufw allow 8501 (Configure firewall if required)

sudo journalctl -u streamlit.service

sudo journalctl -u streamlit.service -f
```

## 4. Finding ExecStart path
```bash
conda activate your_env

which streamlit

/home/pixii/miniconda3/envs/pixii_ai/bin/streamlit
```
