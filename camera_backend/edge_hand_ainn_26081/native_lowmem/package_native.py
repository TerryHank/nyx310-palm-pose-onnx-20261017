"""Create and independently read back a NYX310 native algorithm TAR."""
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
SDK=Path('/mnt/c/Users/user/xwechat_files/<WECHAT_USER>/msg/file/2026-10/AINN_SDK_26.08.1')
EXPECTED={'algo_initialize','algo_ctrl_start','algo_ctrl_stop','algo_push_frame',
          'algo_get_param','algo_set_param','algo_set_output_handle','algo_release'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def command(*args):return subprocess.check_output(args,text=True)
def elf(path):
    data=path.read_bytes()
    assert data[:4]==b'\x7fELF' and data[4:6]==b'\x02\x01'
    assert struct.unpack_from('<H',data,18)[0]==183,'not AArch64 ELF'
    return dict(size=len(data),sha256=hashlib.sha256(data).hexdigest(),architecture='AArch64 ELF64 little-endian')

def main():
    stage=ROOT/'stage'/'nyx310'/'ai_common_app_install'
    host=json.loads((ROOT/'native_validation.json').read_text())
    arm=json.loads((ROOT.parent/'lowmem'/'diagnostics'/'arm_core_validation.json').read_text())
    loader=json.loads((ROOT.parent/'lowmem'/'diagnostics'/'arm_loader.json').read_text())
    raw=json.loads((ROOT/'raw_validation.json').read_text())
    arm_raw=json.loads((ROOT.parent/'lowmem'/'diagnostics'/'arm_raw_validation.json').read_text())
    calibration=json.loads((ROOT.parent/'lowmem'/'calibration.json').read_text())
    model_validation=json.loads((ROOT.parent/'output'/'conversion_report.json').read_text())
    assert host['passed'] and arm['passed'] and loader['arm_loader_passed']
    assert raw['passed'] and arm_raw['passed']
    assert all(m['replay_passed'] for m in model_validation['models'])
    original=SDK/'nyx310'/'ai_common_app_install'
    assert sha(stage/'bin'/'ai_common_app')==sha(original/'bin'/'ai_common_app')
    source_include=SDK/'AINN_SDK'/'2_deploy'/'algLib'/'Include'
    assert all(sha(ROOT/'include'/name)==sha(source_include/name) for name in ('alg_api.h','alg_define.h'))
    names=set(re.findall(r'\b(algo_\w+)$',command('nm','-D','--defined-only',str(stage/'lib'/'libalg.so')),re.M))
    assert names==EXPECTED
    dynamic=command('readelf','-d',str(stage/'lib'/'libalg.so'))
    assert '[$ORIGIN]' in dynamic and 'librknnrt.so' in dynamic and 'mock' not in dynamic
    assert elf(stage/'bin'/'ai_common_app') and elf(stage/'lib'/'libalg.so') and elf(stage/'lib'/'librknnrt.so')
    models=PROJECT/'dist'/'hand_models_ainn_26081'
    assert sha(stage/'model'/'palm.rknn')==sha(models/'palm_detection_mediapipe_2023feb_rv1126b_fp16.rknn')
    assert sha(stage/'model'/'handpose.rknn')==sha(models/'handpose_estimation_mediapipe_2023feb_rv1126b_fp16.rknn')
    config=json.loads((stage/'config'/'config.json').read_text())
    assert config['AINN']['InputFrameType']==['Depth','Color']
    assert config['AINN']['Enable']==1 and config['AINN']['AIModuleMode']=='ContinuousRunMode'
    outputs=PROJECT/'dist'/'upload';outputs.mkdir(exist_ok=True)
    archive=outputs/'ainn_PalmPoseGraspXYZ_LowMem2_Ver26_8_1_nyx310.tar'
    # The manufacturer build.sh uses the same nyx310/ai_common_app_install/
    # root and an uncompressed GNU tar. Keep no absolute paths or symlinks.
    prefix='nyx310/ai_common_app_install'
    manifest={str(p.relative_to(stage)):dict(size=p.stat().st_size,sha256=sha(p))
              for p in sorted(stage.rglob('*')) if p.is_file()}
    epoch=int(datetime.datetime(2026,10,16,tzinfo=datetime.timezone.utc).timestamp())
    def metadata(info):
        info.uid=info.gid=0;info.uname=info.gname='root';info.mtime=epoch
        info.mode=0o755 if info.isdir() or '/bin/' in info.name or '/lib/' in info.name else 0o644
        return info
    with tarfile.open(archive,'w',format=tarfile.GNU_FORMAT) as tar:
        tar.add(stage,arcname=prefix,recursive=True,filter=metadata)
    verified={}
    with tarfile.open(archive,'r:') as tar:
        for member in tar.getmembers():
            assert not member.name.startswith('/') and '..' not in Path(member.name).parts
            assert member.name==prefix or member.name.startswith(prefix+'/')
            assert member.isdir() or member.isfile(),'no symlinks/devices permitted'
            if member.isfile():
                relative=str(Path(member.name).relative_to(prefix));data=tar.extractfile(member).read()
                assert hashlib.sha256(data).hexdigest()==manifest[relative]['sha256']
                assert len(data)==manifest[relative]['size'];verified[relative]=True
                assert 'mock' not in relative and 'replay' not in relative
        assert set(verified)==set(manifest)
    report=dict(
        result='LOWMEM2_LOCAL_VALIDATION_PASSED_CAMERA_MEMORY_AND_JSON_TEST_PENDING',
        sdk='26.08.1',camera_target='nyx310',npu_target='rv1126b',
        package=dict(path=str(archive),size=archive.stat().st_size,sha256=sha(archive),
                     format='uncompressed GNU TAR',root=prefix,files=manifest,archive_readback_passed=True),
        supplied_app_unmodified=True,sdk_headers_unmodified=True,sdk_exports=sorted(names),
        elf={name:elf(stage/name) for name in ('bin/ai_common_app','lib/libalg.so','lib/librknnrt.so')},
        elf_dynamic=dynamic,
        exported_symbols=command('nm','-D','--defined-only',str(stage/'lib'/'libalg.so')),
        runtime_requirement='Manufacturer AINN NYX310 firmware/runtime compatible with SDK 26.08.1. Device compatibility not tested.',
        minimum_symbol_versions=dict(
            custom_lib=command('readelf','--version-info',str(stage/'lib'/'libalg.so')),
            supplied_app=command('readelf','--version-info',str(stage/'bin'/'ai_common_app'))),
        implemented=['two RKNN stages','OpenCV Zoo preprocessing/landmark decoding','nearest OPEN/GRASP/CLOSED',
                     'ROI tracking fallback','palm reference and robust aligned depth median',
                     'camera intrinsic backprojection','hand IDs and temporal smoothing','HOLD with valid:false'],
        depth_window_mm=[50,5000],depth_window_changes_camera_range=False,
        calibration_camera='<DEVICE_SERIAL>',
        local_host_validation=host,local_arm_core_validation=arm,local_arm_loader_validation=loader,
        raw_depth_validation=raw,local_arm_raw_validation=arm_raw,
        sdk_depth_ray_calibration=calibration,
        memory_changes=dict(retained_rgb_max_width=640,latest_frame_slots=2,
                            image_storage_before_bytes=raw['old_two_frame_storage_bytes'],
                            image_storage_after_bytes=raw['new_two_frame_storage_bytes'],
                            ray_table_bytes=raw['calibrated_ray_table_bytes'],
                            warmed_host_RSS_growth_bytes=raw['host_warmed_RSS_growth_bytes'],
                            full_rgb_aligned_depth_requested=False,
                            camera_memory_budget_verified=False),
        model_simulator_validation=model_validation,
        notes=['Native palm NMS uses correct XYXY IoU, fixing the Zoo wrapper XYXY/XYWH mismatch.',
               'C++ crop padding uses BORDER_ISOLATED to exactly match Python crop padding.',
               'Host integration tests use ONNX Runtime as a host-only RKNN API substitute, not actual NPU execution.',
               'AArch64 tests execute CPU preprocessing/classification/depth and ELF loading under QEMU; no NPU initialized.',
               'Synthetic and recorded depths are distinguished in depth test cases. No new camera captures were acquired.',
               'Core keeps at most 32 historical hand IDs to bound embedded memory.'],
        camera_connected_by_this_build=True,camera_calibration_read_only=True,
        camera_parameters_modified=False,camera_upload_executed=False,camera_upgrade_executed=False,
        camera_test_executed=False,end_to_end_camera_fps_verified=False,
    )
    report_path=outputs/'NYX310_HAND_LOWMEM2_VALIDATION.json'
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    manual=outputs/'NYX310_HAND_LOWMEM2_IMPORT.txt'
    shutil.copy2(ROOT/'MANUAL_IMPORT.txt',manual)
    checksum=outputs/'NYX310_HAND_LOWMEM2_SHA256SUMS.txt'
    checksum.write_text(''.join(sha(path)+'  '+path.name+'\n' for path in (archive,report_path,manual)),encoding='utf-8')
    print(json.dumps(dict(package=str(archive),bytes=archive.stat().st_size,
                         sha256=sha(archive),archive_entries_verified=len(verified),
                         report=str(report_path),manual=str(manual)),indent=2),flush=True)

if __name__=='__main__':main()
