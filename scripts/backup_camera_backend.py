"""Back up recovered project files and inspect original camera packages offline."""
import argparse
import ast
import hashlib
import json
import re
import shutil
import struct
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backup-root', type=Path, required=True)
    parser.add_argument('--sdk-zip', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    public = repo / 'camera_backend'
    recovered = args.backup_root / 'recovered_project' / 'edge_hand_ainn_26081'
    source = public / 'edge_hand_ainn_26081'
    source.mkdir(parents=True, exist_ok=True)
    source_files = []
    for old in sorted(recovered.rglob('*')):
        if not old.is_file():
            continue
        relative = old.relative_to(recovered)
        if old.name == 'hand.json':
            continue  # Original device-bound calibration is retained locally.
        new = source / relative
        new.parent.mkdir(parents=True, exist_ok=True)
        data = old.read_text(encoding='utf-8')
        data = re.sub(r'\bGN[0-9A-Z]{10,}\b', '<DEVICE_SERIAL>', data)
        data = re.sub(r'wxid_[A-Za-z0-9_]+', '<WECHAT_USER>', data)
        new.write_text(data, encoding='utf-8', newline='\n')
        if new.suffix == '.py':
            ast.parse(data, filename=str(relative))
        source_files.append(dict(path=new.relative_to(repo).as_posix(), sha256=sha(new.read_bytes()),
                                 history_source_sha256=sha(old.read_bytes()),
                                 redacted=data != old.read_text(encoding='utf-8')))
    licenses = public / 'licenses'
    licenses.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.sdk_zip) as sdk:
        (licenses / 'Scepter_AINN_BSD_3_Clause.txt').write_bytes(sdk.read('AINN_SDK/LICENSE'))
        for variant in ('native', 'native_lowmem', 'native_hagrid', 'native_hagrid_binary'):
            for name in ('alg_api.h', 'alg_define.h'):
                target = source / variant / 'include' / name
                target.write_bytes(sdk.read('AINN_SDK/2_deploy/algLib/Include/' + name))
                source_files.append(dict(path=target.relative_to(repo).as_posix(), sha256=sha(target.read_bytes()),
                                         origin='AINN SDK 26.08.1, BSD 3-Clause'))
        # Needed only for local host compilation; never copied into the public repo.
        headers = args.backup_root / 'sdk_headers'
        headers.mkdir(exist_ok=True)
        for name in sdk.namelist():
            if '/librknn_api/include/' in name and not name.endswith('/'):
                (headers / PurePosixPath(name).name).write_bytes(sdk.read(name))
    packages = []
    for package in sorted((args.backup_root / 'native_packages').glob('*.tar')):
        data = package.read_bytes()
        item = dict(file=package.name, bytes=len(data), sha256=sha(data), members=[])
        with tarfile.open(package) as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():
                    raise ValueError('Unsafe archive member: ' + member.name)
                if not member.isfile():
                    continue
                content = archive.extractfile(member).read()
                relative = '/'.join(path.parts[2:])
                entry = dict(path=relative, bytes=len(content), sha256=sha(content), mode=oct(member.mode))
                if content[:4] == b'\x7fELF':
                    entry['elf_machine'] = struct.unpack_from('<H', content, 18)[0]
                    if entry['elf_machine'] != 183:
                        raise ValueError('Camera binary is not AArch64')
                item['members'].append(entry)
                if relative == 'lib/libalg.so' or relative.startswith('model/') or relative.startswith('licenses/'):
                    output = public / 'artifacts' / package.stem / relative
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(content)
                    if sha(output.read_bytes()) != entry['sha256']:
                        raise ValueError('Extracted artifact hash mismatch')
        required = {'bin/ai_common_app', 'lib/libalg.so', 'lib/librknnrt.so', 'model/palm.rknn',
                    'model/handpose.rknn', 'config/config.json', 'config/hand.json'}
        names = {m['path'] for m in item['members']}
        if not required <= names:
            raise ValueError('Native package is incomplete: ' + package.name)
        packages.append(item)
    report = dict(date='2026-10-26', source_recovery='recorded source patches and explicit variant copy steps',
                  source_bitwise_equivalence_to_lost_directory_proven=False,
                  camera_connected=False, camera_modified=False, packages_repacked=False,
                  raw_native_packages_location='local backup native_packages/ (not public)',
                  public_excludes=['vendor application/runtime', 'device calibration', 'capture data', 'credentials'],
                  source_files=source_files, packages=packages)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    (public / 'backup_manifest.json').write_text(encoded, encoding='utf-8')
    (args.backup_root / 'backup_manifest.json').write_text(encoded, encoding='utf-8')
    pc_models = public / 'pc_models'
    pc_models.mkdir(exist_ok=True)
    for old in (repo / 'verification' / 'models').glob('*.onnx'):
        shutil.copy2(old, pc_models / old.name)
    print(json.dumps(dict(recovered_public_files=len(source_files), original_packages=len(packages),
                          package_bytes=sum(p['bytes'] for p in packages)), indent=2))


if __name__ == '__main__':
    main()
