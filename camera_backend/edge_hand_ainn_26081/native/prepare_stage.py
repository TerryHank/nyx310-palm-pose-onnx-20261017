import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SDK=Path('/mnt/c/Users/user/xwechat_files/<WECHAT_USER>/msg/file/2026-10/AINN_SDK_26.08.1')

def main():
    stage=ROOT/'stage'/'nyx310'/'ai_common_app_install'
    for name in ('bin','lib','model','config'):
        (stage/name).mkdir(parents=True,exist_ok=True)
    app=SDK/'nyx310'/'ai_common_app_install'/'bin'/'ai_common_app'
    shutil.copy2(app,stage/'bin'/'ai_common_app')
    (stage/'bin'/'ai_common_app').chmod(0o755)
    shutil.copy2(SDK/'nyx310'/'ai_common_app_install'/'config'/'config.json',stage/'config'/'config.json')
    shutil.copy2(ROOT/'hand.json',stage/'config'/'hand.json')
    models=ROOT.parent/'output'
    for src,dst in [('palm_detection_mediapipe_2023feb_rv1126b_fp16.rknn','palm.rknn'),
                    ('handpose_estimation_mediapipe_2023feb_rv1126b_fp16.rknn','handpose.rknn')]:
        shutil.copy2(models/src,stage/'model'/dst)
    shutil.copy2(ROOT/'build-arm'/'libalg.so',stage/'lib'/'libalg.so')
    shutil.copy2(ROOT/'sdk'/'3rdparty'/'librknn_api'/'aarch64'/'librknnrt.so',stage/'lib'/'librknnrt.so')
    licenses=stage/'licenses';licenses.mkdir(exist_ok=True)
    project=ROOT.parents[1]
    shutil.copy2(project/'verification'/'vendor'/'opencv_zoo'/'LICENSE',licenses/'OpenCV_Zoo_Apache_2.0.txt')
    shutil.copytree(ROOT/'sdk'/'3rdparty'/'opencv'/'share'/'OpenCV'/'licenses',licenses/'OpenCV_dependencies',dirs_exist_ok=True)
    print(stage)
    (ROOT/'stage_manifest.json').write_text(json.dumps({str(p.relative_to(stage)):dict(
        size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(stage.rglob('*')) if p.is_file()},indent=2))

if __name__=='__main__':main()
