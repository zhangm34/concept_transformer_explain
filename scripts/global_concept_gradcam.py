"""Grad-CAM maps for the Concept Transformer's global concepts on the six cached test examples.

Global concepts are attended only by the ViT class token, so the model has no patch map for them.
This script computes a post-hoc approximation: the gradient of one global concept's attention
weight (mean over heads) with respect to the patch tokens entering the last ViT block, where the
class token last reads the patches. Grad-CAM weights each feature channel by its mean gradient
over patches and keeps the positive part. The maps are an outside approximation of where that
attention weight is sensitive to the image, not something the model computes.

Default mode needs the disjoint run's checkpoint. It checks that the checkpoint matches the cached
examples (hash, predictions and global attention), then writes global_gradcam.npz and
global_gradcam.png beside them. --untrained-head builds the pretrained backbone with a randomly
initialized concept head to test the mechanics only; its output must not be presented.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import textwrap
import time
from unittest.mock import patch

import numpy as np

from visualize_test_explanations import BUNDLE, PROJECT_DIR, pretty_attribute, sha256


def build_network(args):
    """The CVIT network: from the checkpoint, or with an untrained concept head for a mechanics test."""
    import torch
    import timm
    from ctc.ctc_model import CTCModel
    from ctc.vit import cub_cvit

    if args.untrained_head:
        torch.manual_seed(0)
        return cub_cvit(), None
    checkpoint_hash = sha256(args.checkpoint)
    checkpoint = torch.load(str(args.checkpoint), map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    assert hparams['ctc_model'] == 'cub_cvit' and not hparams.get('baseline', False)
    create_model = timm.create_model

    def create_without_pretraining(*positional, **kwargs):
        kwargs['pretrained'] = False
        return create_model(*positional, **kwargs)

    with patch.object(timm, 'create_model', create_without_pretraining):
        model = CTCModel(**hparams)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.model, checkpoint_hash


def global_gradcam(net, x, concepts):
    """Grad-CAM over the 14 x 14 patch grid for each global concept index, from one forward pass."""
    import torch

    store = {}

    def keep(name):
        def hook(module, inputs, output):
            output.retain_grad()
            store[name] = output
        return hook

    handles = [net.feature_extractor.blocks[-1].norm1.register_forward_hook(keep('tokens')),
               net.feature_extractor.norm.register_forward_hook(keep('final'))]
    logits, _, global_attention, _ = net(x)
    for handle in handles:
        handle.remove()
    tokens = store['tokens']
    maps, final_gradient_norms = [], []
    for concept in concepts:
        tokens.grad = None
        store['final'].grad = None
        global_attention[0, 0, concept].backward(retain_graph=True)
        gradients = tokens.grad[0, 1:]  # patch tokens only; position 0 is the class token
        activations = tokens.detach()[0, 1:]
        cam = torch.relu((activations * gradients.mean(0)).sum(-1))
        maps.append(cam.reshape(14, 14).numpy())
        # Global concepts read only the class token, so patch tokens after the last block get no gradient.
        final_gradient_norms.append(float(store['final'].grad[0, 1:].abs().max()))
    return logits.detach()[0], global_attention.detach()[0, 0], np.stack(maps), final_gradient_norms


def render(output_dir, report, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    examples = report['examples']
    top_k = data['concepts'].shape[1]
    figure, axes = plt.subplots(len(examples), top_k + 1, figsize=(3.1 * (top_k + 1), 3.25 * len(examples)))
    for row, sample in enumerate(examples):
        image = data['images'][row]
        ax = axes[row, 0]
        ax.imshow(image)
        ax.set_title('{} · ID {}\nTrue: {}\nPred: {} ({:.1%})'.format(
            'CORRECT' if sample['correct'] else 'WRONG', sample['image_id'], sample['true_class'],
            sample['predicted_class'], sample['predicted_probability']),
            color='#167b4b' if sample['correct'] else '#ba3c35', fontsize=9)
        ax.axis('off')
        for column in range(top_k):
            ax = axes[row, column + 1]
            cam = data['maps'][row, column]
            peak = cam.max()
            ax.imshow(image)
            ax.imshow(cam / peak if peak > 0 else cam, cmap='viridis', vmin=0, vmax=1, alpha=0.6,
                      extent=(-0.5, 223.5, 223.5, -0.5), interpolation='nearest')
            name = report['global_attributes'][int(data['concepts'][row, column])]
            ax.set_title(textwrap.fill(pretty_attribute(name), 26)
                         + '\nattention {:.3f}'.format(data['attention'][row, column]), fontsize=9)
            ax.axis('off')
    figure.suptitle('Global concepts: Grad-CAM of each concept\'s attention weight (post-hoc approximation)\n'
                    + report['title_note'], fontsize=12)
    figure.subplots_adjust(top=0.93, bottom=0.05, hspace=0.55, wspace=0.08)
    colorbar_ax = figure.add_axes([0.3, 0.02, 0.4, 0.008])
    figure.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='viridis'), cax=colorbar_ax, orientation='horizontal',
                    label='Grad-CAM, scaled to each map\'s own maximum')
    figure.savefig(output_dir / 'global_gradcam.png', dpi=130, bbox_inches='tight')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, default=BUNDLE / 'l4-45478721/cub_cvit/CUB2011Parts_expl1.0/binary_mnist_best_ckpt.ckpt')
    parser.add_argument('--data-dir', type=Path, default=PROJECT_DIR / 'CUB-200-2011/CUB_200_2011')
    parser.add_argument('--examples-dir', type=Path, default=PROJECT_DIR / 'explanation_artifacts/disjoint-45478721')
    parser.add_argument('--output-dir', type=Path, default=None, help='defaults to --examples-dir')
    parser.add_argument('--top-k', type=int, default=3, help='most-attended global concepts mapped per image')
    parser.add_argument('--threads', type=int, default=min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    parser.add_argument('--untrained-head', action='store_true',
                        help='mechanics test only: pretrained backbone, random concept head')
    args = parser.parse_args()
    output_dir = args.output_dir or args.examples_dir
    if args.untrained_head and output_dir.resolve() == args.examples_dir.resolve():
        parser.error('--untrained-head output is not a result; pass a separate --output-dir.')
    if not 1 <= args.top_k <= 13:
        parser.error('--top-k must be between 1 and 13.')

    start = time.perf_counter()
    import torch
    sys.path.insert(0, str(PROJECT_DIR / 'concept_transformer'))
    from data.cub2011parts import CUB2011Parts_dataset
    from data.cub2011parts_datamodule import CUB2011Parts

    torch.set_num_threads(args.threads)
    cached = json.loads((args.examples_dir / 'examples.json').read_text())
    with np.load(args.examples_dir / 'selected_examples.npz') as saved:
        cached_arrays = dict(saved)
    net, checkpoint_hash = build_network(args)
    if not args.untrained_head:
        assert checkpoint_hash == cached['checkpoint_sha256'], 'checkpoint differs from the one behind the cached examples'
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)

    transform = CUB2011Parts(data_dir=str(args.data_dir), num_workers=0).test_transform
    dataset = CUB2011Parts_dataset(root=str(args.data_dir), train=False, transform=transform)
    position = {int(image_id): i for i, image_id in enumerate(dataset.data.img_id)}
    concepts, attention, maps, checks = [], [], [], []
    for row, sample in enumerate(cached['examples']):
        x = dataset[position[sample['image_id']]][0].unsqueeze(0).requires_grad_(True)
        chosen = np.argsort(-cached_arrays['global_attention'][row], kind='stable')[:args.top_k]
        logits, weights, cams, final_norms = global_gradcam(net, x, chosen.tolist())
        assert max(final_norms) == 0.0, 'patch tokens after the last block should not reach the global stream'
        attention_error = float(np.abs(weights.numpy() - cached_arrays['global_attention'][row]).max())
        if not args.untrained_head:
            assert int(logits.argmax()) == sample['predicted_label'], sample['image_id']
            assert attention_error < 1e-4, (sample['image_id'], attention_error)
        concepts.append(chosen)
        attention.append(weights.numpy()[chosen])
        maps.append(cams)
        checks.append({'image_id': sample['image_id'], 'global_attention_max_abs_error': attention_error,
                       'map_peaks': [float(cam.max()) for cam in cams]})
        print('Image {}: {} maps · {:.1f}s'.format(sample['image_id'], len(cams), time.perf_counter() - start), flush=True)

    data = {'images': cached_arrays['images'], 'concepts': np.stack(concepts),
            'attention': np.stack(attention), 'maps': np.stack(maps)}
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'mode': 'untrained concept head (mechanics test, not a result)' if args.untrained_head else 'checkpoint',
              'checkpoint_path': None if args.untrained_head else str(args.checkpoint),
              'checkpoint_sha256': checkpoint_hash,
              'method': 'Grad-CAM: d(global attention weight, mean over heads) / d(patch tokens after blocks[-1].norm1); '
                        'channel weights = mean gradient over the 196 patches; map = ReLU(sum over channels)',
              'top_k': args.top_k, 'checks': checks, 'examples': cached['examples'],
              'global_attributes': cached['global_attributes'],
              'title_note': ('MECHANICS TEST: untrained concept head, not a result' if args.untrained_head
                             else 'Disjoint run 45478721 · selected epoch {} · CPU float32'.format(cached['checkpoint_epoch_one_based'])),
              'script_sha256': sha256(Path(__file__))}
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / 'global_gradcam.npz', **{k: v for k, v in data.items() if k != 'images'})
    (output_dir / 'global_gradcam.json').write_text(json.dumps(report, indent=2) + '\n')
    render(output_dir, report, data)
    print('Saved global_gradcam.png/.npz/.json to', output_dir, '· {:.1f}s'.format(time.perf_counter() - start))


if __name__ == '__main__':
    main()
