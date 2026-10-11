"""Render Concept Transformer explanations from a saved checkpoint and test split.

Uses the repository's deterministic evaluation transform. Inference is CPU float32;
no training is performed. Cached figures can be displayed without loading the model.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import textwrap
import time
from unittest.mock import patch

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORK_DIR = Path('/blue/yonghui.wu/zhangm3/concept_transformer_explain')
BUNDLE = WORK_DIR / 'training_runs/disjoint-proportional-seed42-20261010'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pretty_attribute(name):
    return name.replace('has_', '').replace('_', ' ').replace('::', ': ')


def pretty_class(name):
    return name.split('.', 1)[-1].replace('_', ' ')


def render(output_dir):
    """Render static PNG/PDF figures and a local HTML gallery from cached tensors."""
    os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_DIR / '.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from html import escape

    output_dir = Path(output_dir)
    report = json.loads((output_dir / 'examples.json').read_text())
    arrays = np.load(output_dir / 'selected_examples.npz')
    global_names = report['global_attributes']
    spatial_names = report['spatial_attributes']
    all_names = ['G · ' + pretty_attribute(n) for n in global_names]
    all_names += ['S · ' + pretty_attribute(n) for n in spatial_names]
    figures = []
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11})

    def heatmap(ax, i, concept_index):
        ax.imshow(arrays['images'][i])
        weights = arrays['spatial_attention'][i, :, concept_index].reshape(14, 14)
        # Display original patches without smoothing or per-image renormalization.
        ax.imshow(weights, cmap='magma', vmin=0, vmax=1, alpha=0.65,
                  extent=(-0.5, 223.5, 223.5, -0.5), interpolation='nearest')
        targets = arrays['spatial_targets'][i, :, concept_index].reshape(14, 14)
        ys, xs = np.where(targets == 1)
        ax.scatter((xs + 0.5) * 16 - 0.5, (ys + 0.5) * 16 - 0.5,
                   s=35, facecolors='none', edgecolors='cyan', linewidths=1.2)
        ax.set_title(textwrap.fill(pretty_attribute(spatial_names[concept_index]), 28)
                     + '\npeak attention = {:.3f}'.format(weights.max()))
        ax.axis('off')

    for i, sample in enumerate(report['examples']):
        color = '#167b4b' if sample['correct'] else '#ba3c35'
        figure = plt.figure(figsize=(16, 8.6))
        grid = figure.add_gridspec(2, 4, height_ratios=[1, 1.12], hspace=0.52, wspace=0.32)
        ax = figure.add_subplot(grid[0, 0])
        ax.imshow(arrays['images'][i])
        ax.set_title('Input crop · image ID {}\nTrue: {}'.format(sample['image_id'], sample['true_class']))
        ax.axis('off')
        for col, concept in enumerate(sample['displayed_spatial_indices'], start=1):
            heatmap(figure.add_subplot(grid[0, col]), i, concept)

        ax = figure.add_axes([0.17, 0.11, 0.27, 0.32])
        attention = arrays['global_attention'][i]
        top = np.argsort(-attention, kind='stable')[:5]
        labels = [textwrap.fill(pretty_attribute(global_names[j]), 34) for j in top]
        ax.barh(np.arange(5), attention[top], color='#4d76a5')
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.set_xlabel('Attention weight (mean across heads)')
        ax.set_title('Top global concepts · * = present in CUB target')
        for row, concept in enumerate(top):
            marker = ' *' if arrays['global_targets'][i, concept] == 1 else ''
            ax.text(attention[concept] + 0.015, row, '{:.3f}{}'.format(attention[concept], marker), va='center')
        ax.grid(axis='x', alpha=0.15)

        ax = figure.add_axes([0.69, 0.11, 0.29, 0.32])
        contributions = arrays['contributions'][i]
        pred, alternative = sample['predicted_label'], sample['comparison_label']
        margin = contributions[:, pred] - contributions[:, alternative]
        top = np.argsort(-np.abs(margin), kind='stable')[:8]
        values = margin[top].tolist()
        labels = [textwrap.fill(all_names[j], 36) for j in top]
        values += [float(margin.sum() - margin[top].sum()), sample['bias_margin']]
        labels += ['All other concepts (sum)', 'Classifier biases (sum)']
        ax.barh(np.arange(len(values)), values,
                color=['#167b4b' if v >= 0 else '#ba3c35' for v in values])
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Contribution to predicted − comparison logit')
        ax.set_title(textwrap.fill('Why {} over {}?'.format(
            sample['predicted_class'], sample['comparison_class']), 48)
                     + '\nTotal logit margin = {:.3f}'.format(sample['logit_margin']))
        ax.grid(axis='x', alpha=0.15)
        figure.suptitle('{} · predicted: {} ({:.1%}) · true: {}'.format(
            'CORRECT' if sample['correct'] else 'WRONG', sample['predicted_class'],
            sample['predicted_probability'], sample['true_class']), color=color, fontsize=15, y=0.99)
        figure.text(0.5, 0.015,
                     'Heatmaps: fixed 0–1 attention scale on 14×14 patches; cyan rings = CUB target patches. '
                     'G = global; S = spatial. Signed bars use every attention head and learned value/projection weights.',
                     ha='center', fontsize=9)
        figure.subplots_adjust(left=0.17, right=0.98, bottom=0.10, top=0.90)
        stem = '{}_{:02d}_image_{}'.format('correct' if sample['correct'] else 'wrong', i + 1, sample['image_id'])
        figure.savefig(output_dir / (stem + '.png'), dpi=150, bbox_inches='tight')
        figure.savefig(output_dir / (stem + '.pdf'), bbox_inches='tight')
        plt.close(figure)
        sample['figure_png'] = stem + '.png'
        sample['figure_pdf'] = stem + '.pdf'
        figures.append(stem + '.png')

    overview, axes = plt.subplots(2, 3, figsize=(14, 11))
    for i, (ax, sample) in enumerate(zip(axes.flat, report['examples'])):
        concept = sample['displayed_spatial_indices'][0]
        heatmap(ax, i, concept)
        color = '#167b4b' if sample['correct'] else '#ba3c35'
        ax.set_title('{} · ID {}\nTrue: {}\nPred: {} ({:.1%})'.format(
            'CORRECT' if sample['correct'] else 'WRONG', sample['image_id'], sample['true_class'],
            sample['predicted_class'], sample['predicted_probability']), color=color)
        top = sample['top_global_indices'][:2]
        caption = 'Spatial: ' + pretty_attribute(spatial_names[concept])
        caption += '\nGlobal: ' + '; '.join(pretty_attribute(global_names[j]) for j in top)
        ax.text(0.5, -0.025, textwrap.fill(caption.split('\n')[0], 44) + '\n'
                + textwrap.fill(caption.split('\n')[1], 44), transform=ax.transAxes,
                ha='center', va='top', fontsize=9)
    overview.suptitle('Concept Transformer · three correct and three wrong test predictions\n'
                       'Disjoint run 45478721 · selected epoch 28 · CPU float32 inference', fontsize=15)
    overview.subplots_adjust(top=0.87, bottom=0.09, hspace=0.64, wspace=0.27)
    colorbar_ax = overview.add_axes([0.29, 0.027, 0.42, 0.012])
    overview.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='magma'), cax=colorbar_ax,
                      orientation='horizontal', label='Concept attention · cyan rings show annotated target patches')
    overview.savefig(output_dir / 'overview.png', dpi=150, bbox_inches='tight')
    overview.savefig(output_dir / 'overview.pdf', bbox_inches='tight')
    plt.close(overview)
    report['render_script_sha256'] = sha256(Path(__file__))
    (output_dir / 'examples.json').write_text(json.dumps(report, indent=2) + '\n')

    cards = []
    for sample in report['examples']:
        title = '{}: {} → {}'.format('Correct' if sample['correct'] else 'Wrong', sample['true_class'], sample['predicted_class'])
        cards.append('<article><h2>{}</h2><p>Image ID {} · prediction {:.1%} · true-class probability {:.1%}</p>'
                     '<img src="{}" alt="{}"><p><a href="{}">Download PDF</a></p></article>'.format(
                         escape(title), sample['image_id'], sample['predicted_probability'], sample['true_probability'],
                         sample['figure_png'], escape(title), sample['figure_pdf']))
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Concept Transformer test explanations</title>'
    html += '<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 24px;color:#222}img{width:100%}article{border-top:1px solid #ddd;margin-top:32px}a{color:#245fa1}</style>'
    html += '<h1>Concept Transformer test explanations</h1><p>Three correct and three incorrect predictions from the saved disjoint test set. '
    html += 'Checkpoint: run 45478721, selected epoch 28. These examples were recomputed in CPU float32 with the saved weights.</p>'
    html += '<p>Heatmaps show concept attention using a fixed 0–1 scale; cyan rings mark positive CUB spatial targets. '
    html += 'Global bars show attention and mark annotated concepts with *. Signed bars reconstruct the predicted-versus-comparison logit margin: '
    html += 'green supports the prediction; red supports the comparison. For errors the comparison is the true class; for correct predictions it is the runner-up.</p>'
    html += '<p>The three spatial maps are selected by largest peak attention, with different attribute families when possible. '
    html += 'Attention denotes how the model represents an input; the signed contribution also includes learned classifier weights. '
    html += 'An unmarked target does not prove an attribute is absent. These six cases illustrate model behavior and do not estimate explanation quality over the test set.</p>'
    html += '<p><a href="https://openreview.net/pdf?id=kAa9eDS0RdO">Paper: Eq. 2 and Figs. 4–5</a> · '
    html += '<a href="examples.json">Sample details and provenance</a> · <a href="overview.pdf">Overview PDF</a></p>'
    html += '<img src="overview.png" alt="Three correct and three wrong predictions">' + ''.join(cards) + '</html>'
    (output_dir / 'gallery.html').write_text(html)
    return report


def extract(args):
    start = time.perf_counter()
    import torch
    import timm
    from torch.utils.data import DataLoader, Subset

    sys.path.insert(0, str(PROJECT_DIR / 'concept_transformer'))
    from ctc.ctc_model import CTCModel
    from data.cub2011parts import CUB2011Parts_dataset
    from data.cub2011parts_datamodule import CUB2011Parts

    torch.set_num_threads(args.threads)
    manifest = json.loads(args.manifest.read_text())
    ids = manifest['image_ids']
    assert all(not (set(ids[a]) & set(ids[b])) for a, b in [('train', 'test'), ('validation', 'test'), ('train', 'validation')])
    for name, expected in manifest['metadata_sha256'].items():
        assert sha256(args.data_dir / 'CUB_200_2011' / name) == expected, name
    assert len(ids['test']) == manifest['counts']['test']
    print('Split verified; loading checkpoint:', args.checkpoint, flush=True)
    checkpoint_hash = sha256(args.checkpoint)
    checkpoint = torch.load(str(args.checkpoint), map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    assert hparams['ctc_model'] == 'cub_cvit' and not hparams.get('baseline', False)
    epoch = int(checkpoint['epoch']) + 1
    # The checkpoint already contains all backbone weights. Avoid downloading or
    # allocating a second pretrained state dictionary during model construction.
    create_model = timm.create_model

    def create_without_pretraining(*positional, **kwargs):
        kwargs['pretrained'] = False
        return create_model(*positional, **kwargs)

    with patch.object(timm, 'create_model', create_without_pretraining):
        model = CTCModel(**hparams)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    model.eval()
    classifier = model.model.classifier
    assert classifier.n_unsup_concepts == 0
    streams = [classifier.concept_tranformer, classifier.spatial_concept_tranformer]
    concepts = [classifier.concepts, classifier.spatial_concepts]
    betas = []
    full_attention = {}
    hooks = []
    with torch.inference_mode():
        for stream, embeddings in zip(streams, concepts):
            heads = stream.num_heads
            count, dim = embeddings.shape[1:]
            kv = stream.kv(embeddings).reshape(1, count, 2, heads, dim // heads)
            values = kv.permute(2, 0, 3, 1, 4)[1, 0]
            projection = stream.proj.weight.reshape(200, heads, dim // heads)
            betas.append(torch.einsum('hcd,ihd->hci', values, projection))
    bias = sum(stream.proj.bias.detach() for stream in streams)

    def capture(name):
        def hook(module, inputs, outputs):
            full_attention[name] = outputs[1].detach()
        return hook

    for name, stream in zip(['global', 'spatial'], streams):
        hooks.append(stream.register_forward_hook(capture(name)))
    transform = CUB2011Parts(data_dir=str(args.data_dir), num_workers=0).test_transform
    dataset = CUB2011Parts_dataset(root=str(args.data_dir), train=False, transform=transform)
    attribute_map = {int(line.split(maxsplit=1)[0]): line.split(maxsplit=1)[1]
                     for line in (args.data_dir / 'CUB_200_2011/attributes.txt').read_text().splitlines()}
    global_names = [attribute_map[j] for j in dataset.non_spatial_attributes_pos]
    spatial_names = [attribute_map[j] for j in dataset.spatial_attributes_pos]
    positions = {int(image_id): i for i, image_id in enumerate(dataset.data.img_id)}
    candidate_test_indices = np.linspace(0, len(ids['test']) - 1, args.max_candidates, dtype=int).tolist()
    official_positions = [positions[ids['test'][i]] for i in candidate_test_indices]
    loader = DataLoader(Subset(dataset, official_positions), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    selected = {True: [], False: []}
    selected_classes = {True: set(), False: set()}
    scanned = []
    maximum_reconstruction_error = 0.0
    offset = 0
    print('Model loaded; scanning evenly spaced test images for distinct classes.', flush=True)
    with torch.inference_mode():
        for x, global_targets, spatial_targets, labels in loader:
            logits, _, global_attention, spatial_attention = model.model(x)
            contributions = []
            for name, beta in zip(['global', 'spatial'], betas):
                mean_attention = full_attention[name].mean(dim=2)
                contributions.append(torch.einsum('bhc,hci->bci', mean_attention, beta))
            contribution = torch.cat(contributions, dim=1)
            reconstructed = contribution.sum(dim=1) + bias
            error = float((reconstructed - logits).abs().max())
            maximum_reconstruction_error = max(maximum_reconstruction_error, error)
            torch.testing.assert_close(reconstructed, logits, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(global_attention.sum(-1), torch.ones_like(global_attention.sum(-1)))
            torch.testing.assert_close(spatial_attention.sum(-1), torch.ones_like(spatial_attention.sum(-1)))
            probabilities = logits.softmax(dim=-1)
            predictions = logits.argmax(dim=-1)
            for row in range(len(x)):
                test_index = candidate_test_indices[offset + row]
                dataset_index = official_positions[offset + row]
                image_id = int(dataset.data.iloc[dataset_index].img_id)
                assert image_id == ids['test'][test_index]
                target, pred = int(labels[row]), int(predictions[row])
                correct = target == pred
                scanned.append({'test_index': test_index, 'image_id': image_id,
                                'true_label': target, 'predicted_label': pred,
                                'correct': correct, 'predicted_probability': float(probabilities[row, pred])})
                if len(selected[correct]) >= args.per_status or target in selected_classes[correct]:
                    continue
                comparison = int(probabilities[row].topk(2).indices[1]) if correct else target
                global_weights = global_attention[row].squeeze(0).numpy()
                spatial_weights = spatial_attention[row].numpy()
                ranking = np.argsort(-spatial_weights.max(axis=0), kind='stable')
                displayed, families = [], set()
                for concept_index in ranking:
                    family = spatial_names[concept_index].split('::')[0]
                    if family not in families:
                        displayed.append(int(concept_index))
                        families.add(family)
                    if len(displayed) == 3:
                        break
                record = {'image_id': image_id, 'test_index': test_index, 'official_test_index': dataset_index,
                          'filepath': str(dataset.data.iloc[dataset_index].filepath),
                          'true_label': target, 'true_class': pretty_class(dataset.class_names[target]),
                          'predicted_label': pred, 'predicted_class': pretty_class(dataset.class_names[pred]),
                          'correct': correct, 'predicted_probability': float(probabilities[row, pred]),
                          'true_probability': float(probabilities[row, target]),
                          'comparison_label': comparison, 'comparison_class': pretty_class(dataset.class_names[comparison]),
                          'comparison_probability': float(probabilities[row, comparison]),
                          'logit_margin': float(logits[row, pred] - logits[row, comparison]),
                          'bias_margin': float(bias[pred] - bias[comparison]),
                          'displayed_spatial_indices': displayed,
                          'top_global_indices': np.argsort(-global_weights, kind='stable')[:5].tolist()}
                image = x[row].permute(1, 2, 0).numpy()
                image = np.clip(image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
                selected[correct].append((record, image, global_weights, spatial_weights,
                                          global_targets[row].numpy(), spatial_targets[row].numpy(),
                                          contribution[row].numpy(), logits[row].numpy()))
                selected_classes[correct].add(target)
            offset += len(x)
            print('Scanned {} images · selected {} correct / {} wrong · elapsed {:.1f}s'.format(
                offset, len(selected[True]), len(selected[False]), time.perf_counter() - start), flush=True)
            if all(len(selected[status]) == args.per_status for status in [True, False]):
                break
    for hook in hooks:
        hook.remove()
    assert all(len(selected[status]) == args.per_status for status in [True, False]), 'Increase --max-candidates to find enough errors.'
    ordered = selected[True] + selected[False]
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'job_id': '45478721',
              'checkpoint_path': str(args.checkpoint), 'checkpoint_sha256': checkpoint_hash,
              'checkpoint_epoch_one_based': epoch,
              'manifest_path': str(args.manifest), 'manifest_sha256': sha256(args.manifest),
              'split_counts': manifest['counts'], 'inference': 'CPU float32, eval, inference_mode',
              'batch_size': args.batch_size, 'cpu_threads': args.threads,
              'selection_rule': 'First three predictions per correctness status with distinct true classes, scanning {} evenly spaced saved test positions in ascending order; stop after both groups are filled.'.format(args.max_candidates),
              'candidate_test_indices': candidate_test_indices, 'scanned_predictions': scanned,
              'attention_heads': [stream.num_heads for stream in streams],
              'maximum_logit_reconstruction_error': maximum_reconstruction_error,
              'global_attribute_ids': dataset.non_spatial_attributes_pos,
              'spatial_attribute_ids': dataset.spatial_attributes_pos,
              'global_attributes': global_names, 'spatial_attributes': spatial_names,
              'source_hashes': {name: sha256(PROJECT_DIR / name) for name in [
                  'concept_transformer/ctc/ctc.py', 'concept_transformer/ctc/vit.py',
                  'concept_transformer/data/cub2011parts.py', 'concept_transformer/data/cub2011parts_datamodule.py']},
              'script_sha256': sha256(Path(__file__)), 'examples': [item[0] for item in ordered]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'examples.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.output_dir / 'selected_examples.npz',
                        **{name: np.stack([item[i] for item in ordered]) for i, name in enumerate([
                            'images', 'global_attention', 'spatial_attention', 'global_targets',
                            'spatial_targets', 'contributions', 'logits'], start=1)}, bias=bias.numpy())
    return render(args.output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=BUNDLE / 'l4-45478721/cub_cvit/CUB2011Parts_expl1.0/binary_mnist_best_ckpt.ckpt')
    parser.add_argument('--manifest', type=Path, default=BUNDLE / 'split_manifest.json')
    parser.add_argument('--data-dir', type=Path, default=WORK_DIR / 'CUB-200-2011/CUB_200_2011')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_DIR / 'explanation_artifacts/disjoint-45478721')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threads', type=int, default=min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    parser.add_argument('--max-candidates', type=int, default=64)
    parser.add_argument('--per-status', type=int, default=3)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.per_status != 3 or args.max_candidates < 6 or args.batch_size < 1 or args.threads < 1:
        parser.error('Use three examples per status, at least six candidates, and positive batch/thread counts.')
    report = render(args.output_dir) if args.render_only else extract(args)
    print('Saved six explanation figures and gallery:', args.output_dir)
    print('Maximum reconstructed-logit error:', report['maximum_logit_reconstruction_error'])


if __name__ == '__main__':
    main()
