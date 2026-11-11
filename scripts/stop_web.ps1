param(
    [int]$From = 7860,
    [int]$Span = 10,
    [switch]$Force
)

# 兜底停止：控制台被强行关闭后，端口可能仍被占用。
# 用法：powershell -ExecutionPolicy Bypass -File scripts\stop_web.ps1
#       powershell -ExecutionPolicy Bypass -File scripts\stop_web.ps1 -Force

$ErrorActionPreference = 'Stop'
$ports = $From..($From + $Span - 1)

$connections = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $ports -contains $_.LocalPort }

if (-not $connections) {
    Write-Host "端口 $From-$($From + $Span - 1) 上没有服务在监听，无需停止。"
    exit 0
}

$pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique

foreach ($procId in $pids) {
    $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if (-not $proc) { continue }
    Write-Host ("发现进程 {0}（{1}），监听端口 {2}" -f $proc.Id, $proc.ProcessName,
        (($connections | Where-Object { $_.OwningProcess -eq $procId } |
            Select-Object -ExpandProperty LocalPort) -join ', '))
}

if (-not $Force) {
    $answer = Read-Host '确认结束这些进程？(y/N)'
    if ($answer -ne 'y') {
        Write-Host '已取消。'
        exit 0
    }
}

foreach ($procId in $pids) {
    try {
        Stop-Process -Id $procId -Force -ErrorAction Stop
        Write-Host "已结束进程 $procId"
    } catch {
        Write-Host ("结束进程 {0} 失败：{1}" -f $procId, $_.Exception.Message)
    }
}