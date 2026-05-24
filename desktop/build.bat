@echo off
echo === AI Proxy Detector — 编译为独立 EXE ===
echo.

pip install pywebview pyinstaller -q
if %ERRORLEVEL% neq 0 (
    echo 依赖安装失败，请检查 Python 环境
    pause & exit /b 1
)

echo [1/2] 正在编译 (--onefile --windowed) ...
pyinstaller --onefile --windowed ^
    --name "AI-Proxy-Detector" ^
    --add-data "index.html;." ^
    --hidden-import webview ^
    --hidden-import webview.platforms.winforms ^
    --clean ^
    app_desktop.py

if %ERRORLEVEL% neq 0 (
    echo 编译失败
    pause & exit /b 1
)

echo.
echo [2/2] 编译完成！
echo 输出位置: dist\AI-Proxy-Detector.exe
echo.
pause