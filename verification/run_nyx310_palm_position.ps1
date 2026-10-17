param(
    [string]$Ip = '192.168.1.101',
    [ValidateRange(0, 86400)][int]$DurationSec = 0,
    [ValidateRange(0.5, 20)][double]$MapExtent = 1.5,
    [string]$Output,
    [ValidateRange(0.01,0.99)][Nullable[double]]$Confidence,
    [switch]$NoWindow,
    [ValidateSet('PalmPoseGraspXYZ','HagridPalmXYZ','HagridBinaryXYZ')][string]$Algorithm = 'PalmPoseGraspXYZ'
)
$ErrorActionPreference = 'Stop'
if (-not $PSBoundParameters.ContainsKey('Algorithm')) {
    & (Join-Path $PSScriptRoot 'run_nyx310_palm_pose_pc.ps1') -Ip $Ip -DurationSec $DurationSec `
        -MapExtent $MapExtent -Output $Output -NoWindow:$NoWindow
    exit $LASTEXITCODE
}
$Viewer = Join-Path $PSScriptRoot 'python\nyx310_camera_palm_viewer.py'
$Scripts = @($Viewer, (Join-Path $PSScriptRoot 'python\kinect_mediapipe_hand_viewer.py'),
             (Join-Path $PSScriptRoot 'python\nyx310_palm_position_viewer.py'),
             (Join-Path $PSScriptRoot 'python\nyx310_palm_pose_pc_viewer.py'),
             (Join-Path $PSScriptRoot 'python\nyx310_hagrid_viewer.py'),
             (Join-Path $PSScriptRoot 'python\nyx310_yolo_hand_viewer.py'))
$Existing = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $CommandLine = $_.CommandLine
    $CommandLine -and (@($Scripts | Where-Object { $CommandLine.Contains($_) }).Count -gt 0) -and
    $CommandLine -match ('--ip\s+"?' + [regex]::Escape($Ip) + '(?:"|\s|$)')
})
foreach ($Process in $Existing) {
    if ($Process.CommandLine -match '--output\s+(?:"([^"]+)"|(\S+))') {
        $PreviousOutput = if ($Matches[1]) { $Matches[1] } else { $Matches[2] }
        if (Test-Path -LiteralPath $PreviousOutput -PathType Container) {
            [IO.File]::WriteAllText((Join-Path $PreviousOutput 'stop.request'), 'switch to palm ROI')
        }
    }
}
if ($Existing.Count) {
    Write-Host 'Closing the previous camera viewer...'
    $Deadline = [DateTime]::UtcNow.AddSeconds(15)
    do {
        Start-Sleep -Milliseconds 200
        $Remaining = @($Existing | Where-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue })
    } while ($Remaining.Count -and [DateTime]::UtcNow -lt $Deadline)
    if ($Remaining.Count) { throw 'Previous viewer has not released the camera.' }
    Start-Sleep -Milliseconds 500
}
if (-not $Output) {
    $Output = Join-Path $PSScriptRoot ("captures\nyx310-palm-position-" + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
}
$Python = Join-Path $PSScriptRoot '.venv-mediapipe\Scripts\python.exe'
$Arguments = @($Viewer, '--ip', $Ip, '--duration-sec', $DurationSec,
               '--map-extent', $MapExtent, '--output', $Output, '--algorithm', $Algorithm)
if ($NoWindow) { $Arguments += '--no-window' }
if ($null -ne $Confidence) { $Arguments += @('--confidence', $Confidence) }
Write-Host 'Backend: CAMERA AINN only (no local inference / no host depth computation)'
Write-Host 'Capture profile: RGB 800x600, 15 FPS; checked before stream start; ToF unchanged'
Write-Host "Camera JSON output: $Output\camera_results.jsonl"
& $Python @Arguments
exit $LASTEXITCODE
