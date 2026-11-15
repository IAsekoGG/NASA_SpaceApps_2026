# Запуск / зупинка локального сервера сайту Project/web у фоні.
#   powershell -File Project/tools/serve_web.ps1            # запустити (порт 8080, уся домашня мережа)
#   powershell -File Project/tools/serve_web.ps1 -Stop      # зупинити
#   powershell -File Project/tools/serve_web.ps1 -Local     # лише з цього ПК (127.0.0.1)
param([int]$Port = 8080, [switch]$Stop, [switch]$Local)

$web = Join-Path (Split-Path $PSScriptRoot -Parent) 'web'
$pidFile = Join-Path $env:TEMP "kakhovka-web-$Port.pid"

if ($Stop) {
    if (Test-Path $pidFile) {
        $p = Get-Content $pidFile
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
        Remove-Item $pidFile
        "Зупинено (pid $p)"
    } else { "Сервер не запущено цим скриптом" }
    return
}

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { "Порт $Port уже зайнятий"; return }
$bind = if ($Local) { '127.0.0.1' } else { '0.0.0.0' }
$log = Join-Path $env:TEMP "kakhovka-web-$Port.log"
$server = Join-Path $PSScriptRoot 'serve_static.py'
$proc = Start-Process python -ArgumentList "`"$server`"", '--port', $Port, '--bind', $bind, '--dir', "`"$web`"" `
    -WindowStyle Hidden -PassThru -RedirectStandardError $log
$proc.Id | Set-Content $pidFile
Start-Sleep -Seconds 1
"Сайт: http://localhost:$Port"
if (-not $Local) {
    Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -like '192.168.*' -or $_.IPAddress -like '10.*' } |
        ForEach-Object { "У мережі: http://$($_.IPAddress):$Port" }
}
"Журнал: $log; зупинити: powershell -File `"$PSCommandPath`" -Stop"
