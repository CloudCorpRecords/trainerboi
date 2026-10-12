@echo off
set "TARGET=%~dp0START.bat"
set "SHORTCUT=%USERPROFILE%\Desktop\TrainerBoi AI Studio.lnk"
set "WORKDIR=%~dp0"

echo Creating desktop shortcut for TrainerBoi...

powershell -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%TARGET%'; $s.WorkingDirectory = '%WORKDIR%'; $s.Description = 'TrainerBoi - GPU Model Training Studio'; $s.Save()"

if exist "%SHORTCUT%" (
    echo.
    echo [SUCCESS] Shortcut created on your Desktop:
    echo "%SHORTCUT%"
    echo.
) else (
    echo [NOTICE] Could not create desktop shortcut automatically. You can double-click START.bat in this folder.
)
pause
