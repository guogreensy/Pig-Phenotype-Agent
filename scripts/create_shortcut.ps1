param(
    [string]$Name = '猪只表型助手'
)

# 在桌面创建快捷方式：双击即可启动服务并打开网页。
# 用法：powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root '启动助手.bat'

if (-not (Test-Path -LiteralPath $target)) {
    throw "找不到启动脚本：$target"
}

$desktop = [Environment]::GetFolderPath('Desktop')
$link = Join-Path $desktop "$Name.lnk"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($link)
$shortcut.TargetPath = $target
$shortcut.WorkingDirectory = $root
$shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,13"
$shortcut.Description = '启动猪只表型智能分析助手并打开网页'
$shortcut.WindowStyle = 1
$shortcut.Save()

Write-Host "已创建桌面快捷方式：$link"
Write-Host "目标：$target"