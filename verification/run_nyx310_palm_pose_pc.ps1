param(
    [string]$Ip = '192.168.1.101',
    [ValidateRange(0,86400)][int]$DurationSec = 0,
    [ValidateRange(0.5,20)][double]$MapExtent = 1.5,
    [ValidateRange(0.01,0.99)][double]$PalmConfidence = 0.45,
    [ValidateRange(0.01,0.99)][double]$PoseConfidence = 0.6,
    [ValidateRange(1000,65000)][int]$RgbExposureCapUs = 8000,
    [string]$Source,
    [string]$Output,
    [switch]$NoWindow,
    [switch]$RecordRgb
)
$ErrorActionPreference='Stop'
if (-not $Source) {
    $Existing=@(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
        $_.CommandLine -and $_.CommandLine.Contains((Join-Path $PSScriptRoot 'python\nyx310_')) -and
        $_.CommandLine -match ('--ip\s+"?'+[regex]::Escape($Ip)+'(?:"|\s|$)')
    })
    foreach($Process in $Existing) {
        if ($Process.CommandLine -match '--output\s+(?:"([^"]+)"|(\S+))') {
            $PreviousOutput=if($Matches[1]){$Matches[1]}else{$Matches[2]}
            if(Test-Path -LiteralPath $PreviousOutput -PathType Container) {
                [IO.File]::WriteAllText((Join-Path $PreviousOutput 'stop.request'),'switch to PC Palm + Hand Pose')
            }
        }
    }
    if($Existing.Count) {
        $Deadline=[DateTime]::UtcNow.AddSeconds(15)
        do {
            Start-Sleep -Milliseconds 200
            $Remaining=@($Existing | Where-Object {Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue})
        } while($Remaining.Count -and [DateTime]::UtcNow -lt $Deadline)
        if($Remaining.Count){throw 'Previous NYX310 viewer has not released the camera.'}
    }
}
if(-not $Output){$Output=Join-Path $PSScriptRoot ('captures\nyx310-palm-pose-pc-'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff'))}
$Arguments=@((Join-Path $PSScriptRoot 'python\nyx310_palm_pose_pc_viewer.py'),'--ip',$Ip,
    '--duration-sec',$DurationSec,'--map-extent',$MapExtent,'--palm-confidence',$PalmConfidence,
    '--pose-confidence',$PoseConfidence,'--rgb-exposure-cap-us',$RgbExposureCapUs,'--output',$Output)
if($Source){$Arguments+=@('--source',$Source)}
if($NoWindow){$Arguments+='--no-window'}
if($RecordRgb){$Arguments+='--record-rgb'}
Write-Host 'Backend: PC OpenCV Zoo Palm + Hand Pose; ROI/Kalman; OPEN/PINCH/GRIP; ToF'
& (Join-Path $PSScriptRoot '.venv-mediapipe\Scripts\python.exe') @Arguments
exit $LASTEXITCODE
