@echo off
rem Pembungkus: jalankan 02_install_offline.ps1 tanpa terhalang ExecutionPolicy.
rem Contoh: 02_install_offline.bat -Dest C:\gemastik
set "HERE=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%02_install_offline.ps1" %*
pause
