@echo off
title AI Coating Web - Global Tunnel Launcher
echo =====================================================================
echo  AI-Driven Coating Formulation ^& Property Optimization
echo  Public Global HTTPS Tunnel
echo =====================================================================
echo.
echo Starting secure Cloudflare Tunnel to forward local port 5000...
echo Keep this window open while sharing the public link.
echo.
if exist "C:\Program Files (x86)\cloudflared\cloudflared.exe" (
    "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:5000
) else if exist "C:\Program Files\cloudflared\cloudflared.exe" (
    "C:\Program Files\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:5000
) else (
    cloudflared tunnel --url http://127.0.0.1:5000
)
pause
