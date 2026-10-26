"""Assemble the actual vendor app with the HaGRID library and verify every archive member."""
import datetime
import hashlib
import json
import re
import shutil
import struct
import subprocess
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
TASK=ROOT.parent/'hagrid_binary'
SDK=Path('/mnt/c/Users/user/xwechat_files/<WECHAT_USER>/msg/file/2026-10/AINN_SDK_26.08.1')
PYTHON='/home/zhangwnag/nyx310-hand-ainn-20261016/venv/bin/python'
COMPILER='/home/zhangwnag/nyx310-hand-ainn-20261016/toolchain/gcc-arm-13.0-2025.09-x86_64-aarch64-none-linux-gnu/bin/aarch64-buildroot-linux-gnu-g++'
EXPECTED={'algo_initialize','algo_ctrl_start','algo_ctrl_stop','algo_push_frame','algo_get_param','algo_set_param','algo_set_output_handle','algo_release'}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def command(*args):return subprocess.check_output(args,text=True)
def elf(path):
    data=path.read_bytes()
    assert data[:6]==b'\x7fELF\x02\x01' and struct.unpack_from('<H',data,18)[0]==183
    return dict(architecture='AArch64 ELF64',size=len(data),sha256=sha(path))


def main():
    stage=ROOT/'stage/nyx310/ai_common_app_install'
    for name in ['bin','lib','model','config','licenses']:(stage/name).mkdir(parents=True,exist_ok=True)
    original=SDK/'nyx310/ai_common_app_install'
    shutil.copy2(original/'bin/ai_common_app',stage/'bin/ai_common_app')
    (stage/'bin/ai_common_app').chmod(0o755)
    configuration=json.loads((original/'config/config.json').read_text())
    configuration['AINN'].update(InputFrameType=['Depth','Color'],PoweronCapture=False,Enable=1,AIModuleMode='ContinuousRunMode')
    (stage/'config/config.json').write_text(json.dumps(configuration,indent=2))
    shutil.copy2(ROOT/'hand.json',stage/'config/hand.json')
    shutil.copy2(ROOT.parent/'lowmem/depth_rays_xy_float32.bin',stage/'config/depth_rays.bin')
    classes={'0':'GRASP','1':'OPEN'}
    (stage/'config/classes.json').write_text(json.dumps(classes,indent=2))
    for source,destination in [(TASK/'hagrid_rv1126b_fp16.rknn','hagrid.rknn'),
                               (ROOT.parent/'output/palm_detection_mediapipe_2023feb_rv1126b_fp16.rknn','palm.rknn'),
                               (ROOT.parent/'output/handpose_estimation_mediapipe_2023feb_rv1126b_fp16.rknn','handpose.rknn')]:
        shutil.copy2(source,stage/'model'/destination)
    shutil.copy2(ROOT/'build-arm/libalg.so',stage/'lib/libalg.so')
    shutil.copy2(ROOT/'sdk/3rdparty/librknn_api/aarch64/librknnrt.so',stage/'lib/librknnrt.so')
    shutil.copy2(PROJECT/'verification/vendor/opencv_zoo/LICENSE',stage/'licenses/OpenCV_Zoo_Apache_2.0.txt')
    shutil.copytree(ROOT/'sdk/3rdparty/opencv/share/OpenCV/licenses',stage/'licenses/OpenCV_dependencies',dirs_exist_ok=True)
    shutil.copy2(TASK/'HaGRID_LICENSE_en_us.pdf',stage/'licenses/HaGRID_LICENSE_en_us.pdf')
    shutil.copy2(PROJECT/'verification/.venv-yoloe/Lib/site-packages/ultralytics-8.4.157.dist-info/licenses/LICENSE',stage/'licenses/Ultralytics_AGPL_LICENSE.txt')
    host=json.loads((ROOT/'hagrid_validation.json').read_text())
    model=json.loads((TASK/'conversion_report.json').read_text())
    sampled=json.loads((ROOT/'sampled_depth_validation.json').read_text())
    assert host['passed'] and model['passed'] and sampled['passed']
    assert host['integration']['no_id_fields'] and host['integration']['maximum_hands']==1
    assert host['integration']['palm_calls_in_20_frames']==4
    sysroot=command(COMPILER,'-print-sysroot').strip()
    loader=sysroot+'/usr/lib/ld-linux-aarch64.so.1'
    qemu=str(ROOT/'qemu/usr/bin/qemu-aarch64-static')
    arm={}
    arm['hagrid_decode']=json.loads(command(qemu,loader,'--library-path',sysroot+'/usr/lib',str(ROOT/'build-arm/hagrid_probe'),str(ROOT/'hagrid_fixtures.bin')))
    arm['core']=json.loads(command(qemu,loader,'--library-path',sysroot+'/usr/lib',str(ROOT/'build-arm/core_probe'),str(ROOT.parent/'native_lowmem/arm_fixtures.bin')))
    arm['raw_depth']=json.loads(command(qemu,loader,'--library-path',sysroot+'/usr/lib',str(ROOT/'build-arm/raw_probe'),str(ROOT.parent/'native_lowmem/raw_fixtures.bin')))
    arm['loader']=json.loads(command(qemu,loader,'--library-path',str(stage/'lib')+':'+sysroot+'/usr/lib',str(ROOT/'build-arm/abi_probe'),str(stage/'lib/libalg.so')))
    assert arm['hagrid_decode']['passed'] and arm['core']['passed'] and arm['raw_depth']['passed'] and arm['loader']['arm_loader_passed']
    exports=command('nm','-D','--defined-only',str(stage/'lib/libalg.so'))
    assert set(re.findall(r'\b(algo_\w+)$',exports,re.M))==EXPECTED
    dynamic=command('readelf','-d',str(stage/'lib/libalg.so'))
    assert '[$ORIGIN]' in dynamic and 'librknnrt.so' in dynamic and 'mock' not in dynamic
    assert sha(stage/'bin/ai_common_app')==sha(original/'bin/ai_common_app')
    binary_strings=command('strings',str(stage/'lib/libalg.so'))
    assert 'hand_id' not in binary_strings and 'nextId' not in binary_strings
    for name in ('alg_api.h','alg_define.h'):
        assert sha(ROOT/'include'/name)==sha(SDK/'AINN_SDK/2_deploy/algLib/Include'/name)
    elf_files={name:elf(stage/name) for name in ['bin/ai_common_app','lib/libalg.so','lib/librknnrt.so']}
    manifest={str(p.relative_to(stage)):dict(size=p.stat().st_size,sha256=sha(p)) for p in sorted(stage.rglob('*')) if p.is_file()}
    assert set(p.name for p in (stage/'model').iterdir())=={'hagrid.rknn','palm.rknn','handpose.rknn'}
    output=PROJECT/'dist/upload';output.mkdir(exist_ok=True)
    archive=output/'ainn_HagridBinaryXYZ_NoID_Fast2_Ver26_8_1_nyx310.tar'
    prefix='nyx310/ai_common_app_install'
    epoch=int(datetime.datetime(2026,10,17,tzinfo=datetime.timezone.utc).timestamp())
    def metadata(info):
        info.uid=info.gid=0;info.uname=info.gname='root';info.mtime=epoch
        info.mode=0o755 if info.isdir() or '/bin/' in info.name or '/lib/' in info.name else 0o644
        return info
    with tarfile.open(archive,'w',format=tarfile.GNU_FORMAT) as tar:tar.add(stage,arcname=prefix,filter=metadata)
    verified=[]
    with tarfile.open(archive,'r:') as tar:
        for member in tar.getmembers():
            assert not member.name.startswith('/') and '..' not in Path(member.name).parts
            assert member.name==prefix or member.name.startswith(prefix+'/')
            assert member.isdir() or member.isfile()
            if member.isfile():
                name=str(Path(member.name).relative_to(prefix));data=tar.extractfile(member).read()
                assert len(data)==manifest[name]['size'] and hashlib.sha256(data).hexdigest()==manifest[name]['sha256']
                assert 'mock' not in name and 'replay' not in name
                if name=='bin/ai_common_app':assert member.mode==0o755
                verified.append(name)
    assert set(verified)==set(manifest)
    report=dict(result='LOCAL_VALIDATION_PASSED_CAMERA_ACCEPTANCE_PENDING',sdk='26.08.1',camera_target='nyx310',npu_target='rv1126b',
                algorithm='HagridBinaryXYZ',version='26.8.1-binary.2',
                package=dict(windows_path='D:/Workspace/_backups/nyx310-camera-backend-20261026/recovered_project/dist/upload/'+archive.name,
                             bytes=archive.stat().st_size,sha256=sha(archive),format='uncompressed GNU TAR',root=prefix,
                             archive_readback_passed=True,files=manifest),
                supplied_app_unmodified=True,sdk_headers_unmodified=True,elf=elf_files,exported_symbols=exports,elf_dynamic=dynamic,
                source_export=json.loads((TASK/'export_report.json').read_text()),rknn_model_simulator=model,
                palm_pose_simulator=json.loads((ROOT.parent/'output/conversion_report.json').read_text()),
                host_production_pipeline=host,arm_qemu=arm,sampled_depth_validation=sampled,
                configuration=configuration,required_rgb=[800,600],required_capture_fps=15,
                calibration_serial='<DEVICE_SERIAL>',
                receiver='D:/Workspace/_backups/nyx310-camera-backend-20261026/recovered_project/verification/run_nyx310_hagrid_binary_camera.ps1',
                identity_tracking=False,identity_fields_in_compiled_library=False,
                maximum_output_hands=1,output_classes=classes,
                new_source_dir=str(ROOT),
                optimizations=dict(yolo_input_before=[640,480],yolo_input_now=[416,320],
                                   yolo_pixel_fraction=(416*320)/(640*480),
                                   palm_relocation_interval_frames=5,pose_model_every_present_frame=True,
                                   depth_sampling_step=2,dense_fallback=True,
                                   duplicate_iou_threshold=.35,maximum_current_results=1,
                                   phase_timings_in_camera_json=True,target_fps=15,target_fps_verified=False),
                old_duplicate_evidence=dict(capture='verification/captures/hagrid-camera-anonymous-20261017-live/camera_results.jsonl',
                    result_index=540,timestamp_ms=5676909,
                    current_top_labels=['no_gesture','no_gesture'],confidence=[.5293,.6646],
                    both_positions_held=True,
                    explanation='Old library allowed two overlapping predictions to become separate tracks and retain different historical depths; exact overlap cannot be computed because boxes were not in the old camera JSON.'),
                capture_behavior='No active power-on capture requested; receiver sets and verifies RGB800x600/15FPS before stream start; firmware initialization defaults are not changed.',
                calibration_note='Raw ToF ray table from prior read-only SDK calibration; this build made no camera connection.',
                camera_connection_executed=False,camera_upload_executed=False,camera_upgrade_executed=False,camera_test_executed=False,
                actual_npu_peak_memory_verified=False,actual_npu_fps_verified=False,gesture_accuracy_verified=False,
                caveats=['This new version has not been installed; actual camera FPS/memory and accuracy remain unverified.',
                         'Host ONNX transport and QEMU CPU tests are not physical NPU tests.',
                         'The exported head matches the actual PC loader end2end=False branch.',
                         'YOLO input416x320 letterboxes RGB at the original aspect ratio; smaller objects may lose accuracy compared with640x480.',
                         '34 original model scores are grouped into GRASP/OPEN without retraining. Detection confidence is presence confidence, not a calibrated binary probability.',
                         'Only one highest-confidence current hand is returned, even when multiple real hands are visible.',
                         'Short-term ROI reuse and one shared position smoother remain; no identity assignment, matching table or hand history list exists.',
                         'No current detection clears the shared position; invalid depth while a hand is still detected can HOLD the previous position.',
                         'Ambiguous no_gesture/take_picture still establish presence; binary state compares the remaining class groups.',
                         'Depth step2 has synthetic-plane regression; the recorded raw calibration scene had no valid palm ROI, so sparse precision on real hand surfaces is not established.',
                         'No proof of contact/object pickup from a gesture class.'])
    report_path=output/'NYX310_HAGRID_BINARY_FAST2_VALIDATION.json';report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    manual=output/'NYX310_HAGRID_BINARY_FAST2_IMPORT.txt';shutil.copy2(ROOT/'MANUAL_IMPORT.txt',manual)
    checksum=output/'NYX310_HAGRID_BINARY_FAST2_SHA256SUMS.txt'
    checksum.write_text(''.join(sha(path)+'  '+path.name+'\n' for path in [archive,report_path,manual]),encoding='utf-8')
    print(json.dumps(dict(package=str(archive),size=archive.stat().st_size,sha256=sha(archive),verified_files=len(verified),arm=arm),indent=2),flush=True)


if __name__=='__main__':main()
