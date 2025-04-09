#.\gpu_monitor.ps1

while ($true) {
    Clear-Host
    nvidia-smi
    Start-Sleep -Seconds 1
}