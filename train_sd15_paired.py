"""Single-GPU direct U-Net fine-tuning; no LoRA and no editing network.

Input: approved_pairs.jsonl from audit_repairs.py, with prompt/before/after.
Optional preservation pool: original TRAIN GenEval JSONL (--keep-results).
The prompt is the full target description. Edit instructions are retained as
metadata if present, NOT fed as extra network inputs. No change mask is used.

Loss: e_plus + pair_weight * relu(margin - gain) + keep_weight * keep_MSE
 gain = (e_minus - e_plus) - (reference_e_minus - reference_e_plus)
This is an experimental denoising ranking surrogate, not exact likelihood/DPO.

Defaults: 75 optimizer updates, 4 pairs/update, rank unrestricted, LR 1e-6.
Same timestep/noise for both sides; frozen VAE posterior means are cached.
Freeze VAE/text encoder/reference; update ALL U-Net parameters with AdamW.
No flip/crop, no EMA, no resume, no automatic evaluation/test-set selection.
Checkpoints contain the U-Net only (FP32, about 3.4 GB each), no optimizer.
To load: UNet2DConditionModel.from_pretrained('.../checkpoint-20/unet',
 torch_dtype=torch.float16), passed as unet= to the ORIGINAL base pipeline.
Start every ablation from the same base revision, not from the previous LoRA.
"""
import argparse
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_inputs(a):
    pairs = rows(a.pairs)
    assert pairs, 'Empty pairs file'
    seen = set()
    for r in pairs:
        assert r['prompt'].strip()
        if 'approved' in r:
            assert str(r['approved']).lower() == 'true', 'Unapproved row'
        if 'kind' in r:
            assert r['kind'] == 'repair', 'Only reviewed repairs are allowed'
        key = (r.get('prompt_id'), r.get('sample_id'))
        if key != (None, None):
            assert key not in seen, f'Duplicate pair: {key}'
            seen.add(key)
        for field in ('before', 'after'):
            r[field] = str(Path(r[field]).resolve(strict=True))
        assert r['before'] != r['after'], 'Pair refers to the same image'
    keep = []
    if a.keep_weight > 0:
        assert a.keep_results, '--keep-results is required when keep-weight > 0'
        pair_paths = {r[k] for r in pairs for k in ('before', 'after')}
        for r in rows(a.keep_results):
            assert isinstance(r['correct'], bool)
            if not r['correct']:
                continue
            path = str(Path(r['filename']).resolve(strict=True))
            assert path not in pair_paths, 'Preservation pool overlaps paired images'
            m = r.get('metadata', {})
            m = json.loads(m) if isinstance(m, str) else m
            keep.append({'image': path, 'prompt': r.get('prompt') or m['prompt']})
        assert keep, 'No correct original training images for preservation'
    return pairs, keep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pairs', required=True)
    p.add_argument('--lock', required=True, help='Pinned model lock with sd15 config')
    p.add_argument('--out', required=True)
    p.add_argument('--keep-results', help='TRAIN-only original-image GenEval results; never DEV/test')
    p.add_argument('--steps', type=int, default=75)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--pair-weight', type=float, default=0.1)
    p.add_argument('--keep-weight', type=float, default=1.0)
    p.add_argument('--margin', type=float, default=0.01)
    p.add_argument('--save-every', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--precision', choices=['bf16', 'fp16'], default='bf16')
    p.add_argument('--check-only', action='store_true', help='Check paths/approvals and exit without model loading')
    a = p.parse_args()
    assert min(a.steps, a.accum, a.save_every) > 0 and a.lr > 0
    assert min(a.pair_weight, a.keep_weight, a.margin) >= 0
    cfg = json.loads(Path(a.lock).read_text())['sd15']
    assert cfg['pipeline'] == 'StableDiffusionPipeline' and cfg['revision']
    pairs, keep = read_inputs(a)
    print(f'{len(pairs)} pairs; {len(keep)} preservation images; {a.steps} updates', flush=True)
    if a.check_only:
        return

    import gc
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    assert torch.cuda.is_available(), 'Requires a CUDA GPU'
    if a.precision == 'bf16':
        assert torch.cuda.is_bf16_supported(), 'Use --precision fp16 on this GPU'
    dtype = torch.bfloat16 if a.precision == 'bf16' else torch.float16
    out = Path(a.out)
    assert not out.exists() or not any(out.iterdir()), 'Choose a fresh output directory'
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    provenance = dict(arguments=vars(a), model=cfg, pairs_sha256=sha(a.pairs),
                      keep_results_sha256=sha(a.keep_results) if a.keep_results else None,
                      script_sha256=sha(__file__), gpu=torch.cuda.get_device_name(0),
                      packages={k: importlib.metadata.version(k) for k in
                                ['torch', 'diffusers', 'transformers', 'safetensors']})
    (out / 'run.json').write_text(json.dumps(provenance, indent=2))
    (out / 'pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs))
    (out / 'keep.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in keep))
    base = dict(pretrained_model_name_or_path=cfg['repo'], revision=cfg['revision'])
    scheduler = DDPMScheduler.from_pretrained(**base, subfolder='scheduler')
    assert scheduler.config.prediction_type == 'epsilon', 'This experiment is SD1.5 epsilon prediction only'

    # Cache deterministic image latents and prompt embeddings, then free encoders.
    vae = AutoencoderKL.from_pretrained(**base, subfolder='vae', torch_dtype=torch.float32).to('cuda')
    encoder = CLIPTextModel.from_pretrained(**base, subfolder='text_encoder', torch_dtype=torch.float32).to('cuda')
    tokenizer = CLIPTokenizer.from_pretrained(**base, subfolder='tokenizer')
    vae.requires_grad_(False).eval()
    encoder.requires_grad_(False).eval()
    latents, texts, image_hashes = {}, {}, {}
    samples = [(r['before'], r['prompt']) for r in pairs] + [(r['after'], r['prompt']) for r in pairs]
    samples += [(r['image'], r['prompt']) for r in keep]
    with torch.no_grad():
        for i, (path, prompt) in enumerate(samples):
            if path not in latents:
                with Image.open(path) as im:
                    assert im.size == (512, 512), f'Expected 512x512 image: {path}'
                    x = torch.from_numpy(np.array(im.convert('RGB'), copy=True)).permute(2, 0, 1)
                x = (x.float().unsqueeze(0).to('cuda') / 127.5) - 1
                latents[path] = (vae.encode(x).latent_dist.mode() * vae.config.scaling_factor).cpu()
                image_hashes[path] = sha(path)
            if prompt not in texts:
                ids = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors='pt').input_ids.to('cuda')
                assert len(tokenizer(prompt).input_ids) <= tokenizer.model_max_length, 'Target prompt would be truncated'
                texts[prompt] = encoder(ids)[0].cpu()
            if (i + 1) % 25 == 0:
                print(f'Encoded {i+1}/{len(samples)} inputs', flush=True)
    (out / 'image_hashes.json').write_text(json.dumps(image_hashes, indent=2))
    del vae, encoder, tokenizer, x, ids
    gc.collect()
    torch.cuda.empty_cache()

    unet = UNet2DConditionModel.from_pretrained(**base, subfolder='unet', torch_dtype=torch.float32).to('cuda')
    unet.requires_grad_(True).train()
    unet.enable_gradient_checkpointing()
    reference = None
    if a.pair_weight > 0 or a.keep_weight > 0:
        reference = UNet2DConditionModel.from_pretrained(**base, subfolder='unet', torch_dtype=torch.float32).to('cuda')
        reference.requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=a.lr, weight_decay=0.0, foreach=False)
    scaler = torch.amp.GradScaler('cuda', enabled=a.precision == 'fp16')
    # Separate RNGs keep pair order and noise matched across loss ablations.
    order_rng, keep_rng = random.Random(a.seed), random.Random(a.seed + 1)
    pair_noise = torch.Generator('cuda').manual_seed(a.seed + 2)
    keep_noise = torch.Generator('cuda').manual_seed(a.seed + 3)
    order = []

    def predict(model, z, t, c):
        with torch.autocast('cuda', dtype=dtype):
            return model(z, t, encoder_hidden_states=c).sample.float()

    def mse(prediction, target):
        return (prediction.float() - target.float()).square().mean()

    def noise_image(z, generator):
        t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device='cuda', generator=generator)
        noise = torch.randn(z.shape, device='cuda', generator=generator)
        return scheduler.add_noise(z, noise, t), noise, t

    print('Training ALL U-Net weights. No LoRA. Loss logs: losses.jsonl', flush=True)
    with (out / 'losses.jsonl').open('w') as log:
        for step in range(1, a.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            totals = dict(sft=0., pair=0., keep=0., gain=0., error_minus=0., margin_active=0.)
            for _ in range(a.accum):
                if not order:
                    order = list(range(len(pairs)))
                    order_rng.shuffle(order)
                r = pairs[order.pop()]
                c = texts[r['prompt']].to('cuda')
                plus, minus = latents[r['after']].to('cuda'), latents[r['before']].to('cuda')
                zplus, noise, t = noise_image(plus, pair_noise)
                ep = mse(predict(unet, zplus, t, c), noise)
                em = ep.new_zeros(())
                gain, rank = ep.new_zeros(()), ep.new_zeros(())
                if a.pair_weight > 0:
                    zminus = scheduler.add_noise(minus, noise, t)
                    with torch.no_grad():
                        rp = mse(predict(reference, zplus, t, c), noise)
                        rm = mse(predict(reference, zminus, t, c), noise)
                    em = mse(predict(unet, zminus, t, c), noise)
                    gain = (em - ep) - (rm - rp)
                    rank = torch.relu(a.margin - gain)
                scaler.scale((ep + a.pair_weight * rank) / a.accum).backward()
                for key, value in [('sft', ep), ('pair', rank), ('gain', gain), ('error_minus', em)]:
                    totals[key] += value.detach().item() / a.accum
                totals['margin_active'] += float(rank.detach().item() > 0) / a.accum
                # Free paired graphs before a separate preservation forward/backward.
                del ep, em, gain, rank
                if a.keep_weight > 0:
                    k = keep_rng.choice(keep)
                    zk = latents[k['image']].to('cuda')
                    ck = texts[k['prompt']].to('cuda')
                    noisy, _, tk = noise_image(zk, keep_noise)
                    with torch.no_grad():
                        target = predict(reference, noisy, tk, ck)
                    preserve = mse(predict(unet, noisy, tk, ck), target)
                    scaler.scale(a.keep_weight * preserve / a.accum).backward()
                    totals['keep'] += preserve.detach().item() / a.accum
                    del preserve, target
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            record = dict(step=step, **totals, grad_norm=float(norm),
                          peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
            log.write(json.dumps(record)+'\n')
            log.flush()
            print(json.dumps(record), flush=True)
            if step % a.save_every == 0 or step == a.steps:
                dest = out / f'checkpoint-{step}'
                unet.save_pretrained(dest / 'unet', safe_serialization=True)
                (dest / 'base_model.json').write_text(json.dumps(cfg, indent=2))
                print(f'Saved full U-Net: {dest}/unet', flush=True)


if __name__ == '__main__':
    main()
