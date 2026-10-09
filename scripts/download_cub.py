"""Download official CUB images/annotations and place them for the repository loader."""
import argparse
import hashlib
from pathlib import Path
import shutil
import ssl
import tarfile
import urllib.request

PROJECT_DIR = Path(__file__).resolve().parents[1]
URL = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
MD5 = '97eceeb196236b17998738112f37df78'  # Published by CaltechDATA.
REQUIRED = ['images.txt', 'classes.txt', 'image_class_labels.txt', 'train_test_split.txt',
            'bounding_boxes.txt', 'parts/part_locs.txt', 'attributes/image_attribute_labels.txt']

def verify(root):
    dataset = root / 'CUB_200_2011'
    missing = [name for name in REQUIRED + ['attributes.txt'] if not (dataset / name).is_file()]
    if missing:
        raise RuntimeError('Missing annotations: ' + ', '.join(missing))
    lines = (dataset / 'images.txt').read_text().splitlines()
    assert len(lines) == 11788, 'Expected 11,788 referenced images.'
    missing_images = [line.split(maxsplit=1)[1] for line in lines
                      if not (dataset / 'images' / line.split(maxsplit=1)[1]).is_file()]
    if missing_images:
        raise RuntimeError('Missing image files: ' + ', '.join(missing_images[:5]))
    print('Verified 11,788 referenced images and required annotation files.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_DIR / 'CUB-200-2011' / 'CUB_200_2011',
                        help='Loader root, the parent of the inner CUB_200_2011 directory.')
    parser.add_argument('--check-only', action='store_true', help='Verify local data without downloading or writing.')
    parser.add_argument('--archive', type=Path, help='Use an already downloaded official .tgz archive.')
    args = parser.parse_args()
    root = args.data_dir.resolve()
    dataset = root / 'CUB_200_2011'
    if args.check_only:
        verify(root)
        return
    if not all((dataset / name).is_file() for name in REQUIRED):
        archive = args.archive or PROJECT_DIR / '.cache' / 'downloads' / 'CUB_200_2011.tgz'
        archive = archive.resolve()
        if not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            part = archive.with_suffix('.tgz.part')
            try:
                import certifi
                context = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                context = ssl.create_default_context()
            print('Downloading official archive (approximately 1.2 GB):', URL, flush=True)
            request = urllib.request.Request(URL, headers={'User-Agent': 'concept-transformer-assessment'})
            with urllib.request.urlopen(request, context=context) as response, part.open('wb') as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            part.replace(archive)
        checksum = hashlib.md5()
        with archive.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                checksum.update(block)
        if checksum.hexdigest() != MD5:
            raise RuntimeError('Archive does not match the MD5 published by CaltechDATA; extraction skipped.')
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(archive), 'r:gz') as bundle:
            members = bundle.getmembers()
            for member in members:
                destination = (root / member.name).resolve()
                if root != destination and root not in destination.parents:
                    raise RuntimeError('Unsafe archive path: ' + member.name)
                if not (member.isfile() or member.isdir()):
                    raise RuntimeError('Unsupported archive member: ' + member.name)
            bundle.extractall(str(root), members=members)
        print('Extracted official archive into:', root)
    source, destination = root / 'attributes.txt', dataset / 'attributes.txt'
    if not destination.exists() and source.is_file():
        shutil.copy2(source, destination)
        print('Copied attribute definitions into:', destination)
    verify(root)
    print('Source data is ready. The notebook generates bounding-box crops with the repository code.')
    print('Segmentation masks are optional and unused by this model; this script does not download them.')


if __name__ == '__main__':
    main()
