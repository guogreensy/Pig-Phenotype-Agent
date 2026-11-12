param(
    [string]$Name = '猪只表型助手',
    [switch]$Desktop
)

# 在项目根目录创建快捷方式：双击即启动服务并打开网页。
# 默认放在项目目录内（便于随工程一起拷贝），加 -Desktop 才放到桌面。
# 用法：powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root '启动助手.bat'

if (-not (Test-Path -LiteralPath $target)) {
    throw "找不到启动脚本：$target"
}

if ($Desktop) {
    $link = Join-Path ([Environment]::GetFolderPath('Desktop')) "$Name.lnk"
} else {
    $link = Join-Path $root "$Name.lnk"
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($link)
$shortcut.TargetPath = $target
$shortcut.WorkingDirectory = $root
$shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,13"
$shortcut.Description = '启动猪只表型智能分析助手并打开网页'
$shortcut.WindowStyle = 1
$shortcut.Save()

Write-Host "已创建快捷方式：$link"
Write-Host "目标：$target"
Write-Host "工作目录：$root"