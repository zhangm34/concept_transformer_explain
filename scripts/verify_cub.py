"""Full source-data audit for the downloaded official CUB-200-2011 archive."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'CUB-200-2011' / 'CUB_200_2011'
DATASET = ROOT / 'CUB_200_2011'
ARCHIVE = PROJECT / '.cache' / 'downloads' / 'CUB_200_2011.tgz'
OUTPUT = PROJECT / 'setup_logs' / 'cub-download.json'
PUBLISHED_MD5 = '97eceeb196236b17998738112f37df78'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def hashes(path):
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            md5.update(block)
            sha256.update(block)
    return {'md5': md5.hexdigest(), 'sha256': sha256.hexdigest()}


def indexed(name, width):
    rows = {}
    for line_number, line in enumerate((DATASET / name).read_text().splitlines(), 1):
        fields = line.split(maxsplit=width - 1)
        assert len(fields) == width, (name, line_number, fields)
        row_id = int(fields[0])
        assert row_id not in rows, (name, 'duplicate ID', row_id)
        rows[row_id] = fields[1:]
    return rows


def main():
    start = time.monotonic()
    report = {
        'status': 'running', 'audit_started_utc': utc_now(),
        'source_url': 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1',
        'checksum_source_url': 'https://data.caltech.edu/records/65de6-vp158',
        'archive_path': str(ARCHIVE), 'loader_root': str(ROOT),
        'dataset_path': str(DATASET), 'published_md5': PUBLISHED_MD5,
        'archive_bytes': ARCHIVE.stat().st_size,
        'archive_download_completed_utc': datetime.fromtimestamp(ARCHIVE.stat().st_mtime, timezone.utc).isoformat(),
    }
    try:
        print('Audit started:', report['audit_started_utc'], flush=True)
        report['archive_hashes'] = hashes(ARCHIVE)
        assert report['archive_hashes']['md5'] == PUBLISHED_MD5
        report['archive_md5_matches_published'] = True
        print('Published archive MD5 verified.', flush=True)
        image_ids = set(range(1, 11789))
        images = indexed('images.txt', 2)
        classes = indexed('classes.txt', 2)
        labels = indexed('image_class_labels.txt', 2)
        splits = indexed('train_test_split.txt', 2)
        boxes = indexed('bounding_boxes.txt', 5)
        attributes = indexed('attributes.txt', 2)
        parts = indexed('parts/parts.txt', 2)
        for name, rows in [('images', images), ('labels', labels), ('splits', splits), ('boxes', boxes)]:
            assert set(rows) == image_ids, name
        assert set(classes) == set(range(1, 201))
        assert set(attributes) == set(range(1, 313))
        assert set(parts) == set(range(1, 16))
        assert len({value[0] for value in images.values()}) == 11788
        assert all(int(row[0]) in classes for row in labels.values())
        assert all(row[0] in {'0', '1'} for row in splits.values())
        split_counts = Counter(int(value[0]) for value in splits.values())
        assert split_counts == {1: 5994, 0: 5794}, split_counts
        class_counts = Counter(int(value[0]) for value in labels.values())
        assert set(class_counts) == set(range(1, 201))
        per_class_split = Counter((int(labels[k][0]), int(splits[k][0])) for k in images)
        assert all(per_class_split[(class_id, split)] > 0 for class_id in classes for split in (0, 1))
        annotation_names = ['images.txt', 'classes.txt', 'image_class_labels.txt',
                            'train_test_split.txt', 'bounding_boxes.txt', 'attributes.txt',
                            'parts/parts.txt', 'parts/part_locs.txt',
                            'attributes/image_attribute_labels.txt',
                            'attributes/class_attribute_labels_continuous.txt']
        report['annotation_sha256'] = {name: hashes(DATASET / name)['sha256'] for name in annotation_names}
        assert (DATASET / 'attributes.txt').read_bytes() == (ROOT / 'attributes.txt').read_bytes()
        report['attribute_definitions_copied_and_equal'] = True
        dimensions = {}
        image_manifest = hashlib.sha256()
        image_modes = Counter()
        boundary_examples = []
        boundary_count = 0
        for count, image_id in enumerate(sorted(images), 1):
            relative = images[image_id][0]
            assert Path(relative).parts[0] == classes[int(labels[image_id][0])][0]
            path = DATASET / 'images' / relative
            assert path.is_file(), path
            raw_sha = hashes(path)['sha256']
            image_manifest.update(f'{image_id} {relative} {raw_sha}\n'.encode())
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                assert image.format == 'JPEG', path
                width, height = image.size
                assert width > 0 and height > 0
                dimensions[image_id] = image.size
                image_modes[image.mode] += 1
                image.convert('RGB').load()
            x, y, w, h = map(float, boxes[image_id])
            assert all(math.isfinite(v) for v in (x, y, w, h))
            assert x >= 0 and y >= 0 and w > 0 and h > 0
            assert round(w) > 0 and round(h) > 0
            if x + w > width + 1.01 or y + h > height + 1.01:
                boundary_count += 1
                if len(boundary_examples) < 10:
                    boundary_examples.append({'image_id': image_id, 'bbox': [x, y, w, h], 'size': [width, height]})
            if count % 2000 == 0:
                print(f'Image content verified and decoded: {count}/11788', flush=True)
        actual_images = {str(path.relative_to(DATASET / 'images')) for path in (DATASET / 'images').rglob('*.jpg')}
        assert actual_images == {row[0] for row in images.values()}
        report['images'] = {
            'referenced': len(images), 'files': len(actual_images), 'fully_decoded': len(dimensions),
            'pil_verify_passed': len(dimensions), 'rgb_conversion_passed': len(dimensions),
            'modes': dict(image_modes), 'ordered_content_manifest_sha256': image_manifest.hexdigest(),
            'bounding_boxes_positive_finite': len(boxes),
            'bounding_boxes_beyond_image_plus_one_pixel': boundary_count,
            'bounding_box_boundary_examples': boundary_examples,
        }
        part_masks = {image_id: 0 for image_id in images}
        visibility = Counter()
        part_rows = 0
        with (DATASET / 'parts/part_locs.txt').open() as handle:
            for line in handle:
                fields = line.split()
                assert len(fields) == 5
                image_id, part_id = map(int, fields[:2])
                x, y = map(float, fields[2:4])
                visible = int(fields[4])
                assert image_id in image_ids and part_id in parts
                assert visible in (0, 1)
                assert math.isfinite(x) and math.isfinite(y)
                assert not part_masks[image_id] & (1 << (part_id - 1))
                part_masks[image_id] |= 1 << (part_id - 1)
                visibility[visible] += 1
                part_rows += 1
        assert part_rows == 11788 * 15
        assert all(mask == (1 << 15) - 1 for mask in part_masks.values())
        report['parts'] = {'definitions': len(parts), 'rows': part_rows, 'unique_image_part_pairs': part_rows,
                           'all_images_have_all_15_parts': True, 'visibility_counts': dict(visibility)}
        attr_masks = {image_id: 0 for image_id in images}
        attr_rows = 0
        presence = Counter()
        certainty = Counter()
        times_negative = 0
        surplus_rows = 0
        surplus_image_ids = Counter()
        surplus_examples = []
        with (DATASET / 'attributes/image_attribute_labels.txt').open() as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.split()
                assert len(fields) in (5, 6), (attr_rows, fields)
                if len(fields) == 6:
                    assert fields[4] == '0', (line_number, fields)
                    surplus_rows += 1
                    surplus_image_ids[int(fields[0])] += 1
                    if len(surplus_examples) < 5:
                        surplus_examples.append({'line': line_number, 'fields': fields})
                image_id, attr_id, present, confidence = map(int, fields[:4])
                annotation_time = float(fields[-1])
                assert image_id in image_ids and attr_id in attributes
                assert present in (0, 1) and confidence in (1, 2, 3, 4)
                assert math.isfinite(annotation_time)
                assert not attr_masks[image_id] & (1 << (attr_id - 1)), (image_id, attr_id)
                attr_masks[image_id] |= 1 << (attr_id - 1)
                presence[present] += 1
                certainty[confidence] += 1
                times_negative += annotation_time < 0
                attr_rows += 1
        assert attr_rows == 11788 * 312
        assert all(mask == (1 << 312) - 1 for mask in attr_masks.values())
        continuous_rows = (DATASET / 'attributes/class_attribute_labels_continuous.txt').read_text().splitlines()
        assert len(continuous_rows) == 200
        for row in continuous_rows:
            values = list(map(float, row.split()))
            assert len(values) == 312
            assert all(math.isfinite(value) and 0 <= value <= 100 for value in values)
        report['attributes'] = {'definitions': len(attributes), 'rows': attr_rows,
                                'unique_image_attribute_pairs': attr_rows,
                                'all_images_have_all_312_attributes': True,
                                'presence_counts': dict(presence), 'certainty_counts': dict(certainty),
                                'negative_annotation_times': times_negative,
                                'source_rows_with_surplus_zero_token': surplus_rows,
                                'surplus_token_image_id_counts': dict(surplus_image_ids),
                                'surplus_token_examples': surplus_examples,
                                'source_annotation_file_preserved': True,
                                'audit_time_field': 'Last whitespace-separated token; surplus zero is reported, never rewritten.',
                                'class_attribute_matrix_shape': [200, 312],
                                'class_attribute_percentages_in_range': True}
        report['split_counts'] = {'train': split_counts[1], 'test': split_counts[0]}
        report['classes'] = {'count': len(classes), 'min_images': min(class_counts.values()),
                             'max_images': max(class_counts.values()), 'all_classes_in_both_splits': True,
                             'image_folder_names_match_labels': True}
        report['source_irregularities'] = [
            f'{surplus_rows} official attribute rows contain a surplus zero token after certainty; image/attribute/presence fields remain valid and source annotations are preserved.',
            f'{boundary_count} official bounding boxes extend beyond image bounds by more than one pixel; repository Pillow cropping preserves its original padding behavior.',
        ]
        report['checks'] = [
            'Official archive matches CaltechDATA published MD5.',
            'All annotation files exist and SHA256 hashes are recorded.',
            'Image IDs are unique and exactly 1..11788 across image/label/split/bbox tables.',
            'All 11788 JPEG files pass PIL.verify, complete decoding, and RGB conversion.',
            'Images directory matches images.txt without missing or extra JPEGs.',
            'All class labels and image directory names agree; all classes appear in both splits.',
            '5994 training and 5794 test images; splits are binary.',
            'All bounding boxes are finite, nonnegative, and have positive dimensions.',
            'Every image has one row for each of 15 parts; visibility is binary and coordinates finite.',
            'Every image has one row for each of 312 attributes; presence binary, certainty 1..4, time finite.',
            'Class attribute percentages form a finite 200x312 matrix in [0,100].',
            'Inner attributes.txt matches the official outer definitions.',
        ]
        report['status'] = 'passed'
        print('Full source image and annotation audit passed.', flush=True)
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        report['audit_completed_utc'] = utc_now()
        report['duration_seconds'] = round(time.monotonic() - start, 3)
        OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
        print('Audit report:', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
