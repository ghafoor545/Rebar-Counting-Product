Rebar Counting App - Pi Quick Guide
1. Start the app manually
cd ~/Desktop/Rebar-Counting-Product
./run-rebar.sh

Backend + frontend start honge

Chromium automatically fullscreen me frontend open karega

2. Start via systemd service

Pehle ensure rebar.service enabled hai:

sudo systemctl enable rebar.service   # autostart on reboot
sudo systemctl start rebar.service    # immediately start
3. Stop the app
sudo systemctl stop rebar.service

Backend + frontend + Chromium band ho jayega

4. Disable autostart (if needed)
sudo systemctl disable rebar.service

Pi reboot ke baad app automatically start nahi hoga

5. Check app status
sudo systemctl status rebar.service

active (running) → app running

inactive (dead) → app stopped

6. Open terminal (if Chromium fullscreen)

GUI terminal: Ctrl + Alt + T

Virtual console: Ctrl + Alt + F2 (wapas GUI: Ctrl + Alt + F7)