$ErrorActionPreference='Stop'
$Destination=Join-Path (Split-Path $PSScriptRoot) 'verification\models'
New-Item -ItemType Directory -Path $Destination -Force | Out-Null
$Models=@(
    @{Repo='palm_detection_mediapipe'; Name='palm_detection_mediapipe_2023feb.onnx'; Sha='78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c'},
    @{Repo='handpose_estimation_mediapipe'; Name='handpose_estimation_mediapipe_2023feb.onnx'; Sha='db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c'}
)
foreach($Model in $Models) {
    $Target=Join-Path $Destination $Model.Name
    if(-not (Test-Path -LiteralPath $Target)) {
        Invoke-WebRequest ('https://huggingface.co/opencv/'+$Model.Repo+'/resolve/main/'+$Model.Name) -OutFile $Target
    }
    if((Get-FileHash $Target -Algorithm SHA256).Hash.ToLower() -ne $Model.Sha) {
        throw ('Model SHA-256 mismatch: '+$Model.Name)
    }
    Write-Host ('Verified '+$Model.Name)
}
