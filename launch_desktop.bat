@echo off
cd /d %~dp0desktop
if not exist node_modules call npm install
call npm start
