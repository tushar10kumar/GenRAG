@echo off
cd /d "%~dp0"
call env\Scripts\activate
python gui_app.py
pause