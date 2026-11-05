"""Prepare reviewed HQ-Edit pairs for train_sd15_paired.py (no GPU required).

prepare: stream a bounded number of rows, filter candidate edits, save 512px
images and review.csv plus paginated HTML. These are candidates, NOT verified.
export: accept explicit TRUE only, validate target prompt length, deduplicate,
exclude exact held-out captions, and export up to --target pairs for training.

Uses original target descriptions, never the edit instruction as the prompt.
Near-square images only; full-frame resize without cropping/flipping.
No automatic GenEval claims: images lack GenEval object/relation annotations.
License: HQ-Edit CC-BY-NC-4.0; retain dataset attribution in research artifacts.
"""
import argparse
import csv
import hashlib
import html
import io
import json
import random
import re
from collections import Counter
from pathlib import Path

REPO = 'UCSC-VLAA/HQ-Edit'
COLORS = r'\b(red|blue|green|yellow|orange|purple|pink|brown|black|white)\b'
OBJECTS = r'\b(person|people|man|woman|boy|girl|dog|cat|bird|horse|cow|sheep|elephant|bear|zebra|giraffe|car|bus|truck|bicycle|motorcycle|boat|chair|bench|table|cup|bowl|bottle|vase|clock|book|banana|apple|orange|backpack|umbrella|handbag|suitcase|bed|couch|toilet|tv|laptop|keyboard|phone|refrigerator|oven|sink|toothbrush|skateboard|surfboard|tie)s?\b'
BROAD = r'\b(background|style|lighting|weather|sunset|season|watercolor|cartoon|anime|hair|expression|zoom|panoramic)\b'


def normalize(s):
    return re.sub(r'[^a-z0-9 ]', '', ' '.join(s.lower().split()))


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def exclusions(paths):
    return {normalize(r['prompt']) for p in paths for r in read_rows(p)}


def tag_edit(edit):
    text = edit.lower()
    if re.search(BROAD, text) or not re.search(OBJECTS, text):
        return None
    if re.search(r'\b(left|right|above|below)\b', text):
        return 'position_candidate'
    if re.search(COLORS, text) and re.search(r'\b(change|turn|make|recolor|colour|color)\b', text):
        return 'color_candidate'
    if re.search(r'\b(add|remove|delete|insert|two|three|four|2|3|4)\b', text):
        return 'presence_count_candidate'
    return None


def decode_image(value):
    from PIL import Image, ImageOps

    if isinstance(value, dict):
        value = value.get("bytes") or value.get("path")

    if isinstance(value, (bytes, bytearray, memoryview)):
        value = Image.open(io.BytesIO(bytes(value)))
    elif isinstance(value, (str, Path)):
        value = Image.open(value)

    return ImageOps.exif_transpose(value).convert("RGB")


def fingerprint(im):
    # Normalized pixel hash catches exact duplicates, NOT semantic near-duplicates.
    return hashlib.sha256(im.resize((64, 64)).tobytes()).hexdigest()


def gallery(out, rows):
    pages = []
    for start in range(0, len(rows), 50):
        name = f'review_{start//50:03d}.html'
        pages.append(name)
        cards = []
        for r in rows[start:start+50]:
            imgs = ''.join(f'<figure><figcaption>{side}</figcaption><img loading="lazy" src="{Path(r[side]).relative_to(out).as_posix()}"></figure>' for side in ['before','after'])
            cards.append(f'<section><h3>{r["id"]}: {r["tag"]}</h3><p>Edit: {html.escape(r["instruction"])}</p><p>Target: {html.escape(r["prompt"])}</p><div>{imgs}</div></section>')
        (out/name).write_text('<meta charset="utf-8"><style>body{font:16px sans-serif;margin:24px}div{display:flex;flex-wrap:wrap}figure{margin:6px}img{width:384px;max-width:100%}section{border-bottom:1px solid #ccc}</style>'+''.join(cards))
    (out/'index.html').write_text('<h1>HQ-Edit candidate review</h1><p>Label review.csv. Blank is excluded. Approve only if the target describes the after image, the before image violates the intended requirement, and quality is acceptable. Inspect backgrounds and duplicate scenes.</p>'+''.join(f'<p><a href="{n}">{n}</a></p>' for n in pages))


def prepare(a):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from PIL import Image
    from transformers import CLIPTokenizer
    assert not a.out.exists(), 'Use a fresh output directory; existing reviews are preserved'
    blocked = exclusions(a.exclude)
    tokenizer = CLIPTokenizer.from_pretrained(a.model, subfolder='tokenizer', revision=a.model_revision)
    api = HfApi()
    revision = api.dataset_info(REPO, revision=a.revision).sha
    files = sorted(p for p in api.list_repo_files(REPO, repo_type='dataset', revision=revision) if p.endswith('.parquet'))
    random.Random(a.seed).shuffle(files)  # Stream randomized shards, not image-heavy buffered shuffle.
    urls = [f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{p}' for p in files]
    ds = load_dataset('parquet', data_files={'train': urls}, split='train', streaming=True)
    a.out.mkdir(parents=True)
    config = dict(repo=REPO, revision=revision, seed=a.seed, max_scan=a.max_scan,
                  candidates=a.candidates, max_aspect=a.max_aspect,
                  tokenizer_model=a.model, tokenizer_revision=a.model_revision,
                  excluded_files={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in a.exclude},
                  note='Keyword-filtered candidates, not GenEval verified. No crop; full-frame resize.')
    (a.out/'source.json').write_text(json.dumps(config, indent=2))
    selected, seen, counts = [], set(), Counter()
    # JSONL is written incrementally so completed candidate rows survive interruption.
    with (a.out/'candidates.jsonl').open('w') as sink:
        for index, row in enumerate(ds):
            if index >= a.max_scan or len(selected) >= a.candidates:
                break
            counts['scanned'] += 1
            prompt, instruction = str(row['output']).strip(), str(row['edit']).strip()
            tag = tag_edit(instruction)
            if not tag or not prompt or normalize(prompt) in blocked:
                continue
            # Remove obvious caption-copy/no-op candidates, but do not infer correctness.
            if normalize(str(row['input'])) == normalize(prompt):
                continue
            before, after = decode_image(row['input_image']), decode_image(row['output_image'])
            if any(max(im.size)/min(im.size) > a.max_aspect for im in [before, after]):
                counts['aspect_rejected'] += 1
                continue
            h0, h1 = fingerprint(before), fingerprint(after)
            if h0 == h1 or h0 in seen or h1 in seen:
                counts['duplicate_or_noop'] += 1
                continue
            seen.update([h0, h1])
            sid = f'{len(selected):05d}'
            folder = a.out/'images'/sid
            folder.mkdir(parents=True)
            for name, im in [('before',before),('after',after)]:
                im.resize((512,512), Image.Resampling.LANCZOS).save(folder/f'{name}.png')
            tokens = len(tokenizer(prompt).input_ids)
            r = dict(id=sid, source_row=index, source_input=str(row['input']),
                     prompt=prompt, instruction=instruction, tag=tag,
                     before=str((folder/'before.png').resolve()), after=str((folder/'after.png').resolve()),
                     before_hash=h0, after_hash=h1, tokens=tokens,
                     needs_shorter_prompt=tokens > tokenizer.model_max_length,
                     approved='', notes='')
            selected.append(r)
            sink.write(json.dumps(r)+'\n'); sink.flush()
            if len(selected) % 25 == 0:
                print(f'Scanned {index+1}; saved {len(selected)} candidates', flush=True)
    assert selected, 'No candidates found within the scan budget'
    write_review(a.out, selected)
    summary = dict(counts, candidates=len(selected), tags=dict(Counter(r['tag'] for r in selected)))
    (a.out/'selection_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('Open index.html; label review.csv; shorten overlong target prompts faithfully.')


def write_review(out, selected):
    with (out/'review.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(selected[0]))
        w.writeheader(); w.writerows(selected)
    gallery(out, selected)


def export(a):
    from transformers import CLIPTokenizer
    from PIL import Image
    source = json.loads((a.out/'source.json').read_text())
    tokenizer = CLIPTokenizer.from_pretrained(source['tokenizer_model'], subfolder='tokenizer', revision=source['tokenizer_revision'])
    blocked = exclusions(a.exclude)
    for name, digest in source['excluded_files'].items():
        p = Path(name)
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, f'Exclusion file changed: {p}'
        blocked |= exclusions([p])
    candidates = {r['id']: r for r in read_rows(a.out/'candidates.jsonl')}
    with (a.out/'review.csv').open(newline='') as f:
        approved = [r for r in csv.DictReader(f) if r['approved'].strip().lower() == 'true']
    random.Random(a.seed).shuffle(approved)
    exported, seen, ids = [], set(), set()
    for r in approved:
        assert r['id'] not in ids, 'Duplicate review ID'
        ids.add(r['id'])
        original = candidates[r['id']]
        prompt = r['prompt'].strip()
        assert prompt and len(tokenizer(prompt).input_ids) <= tokenizer.model_max_length, f'Shorten target prompt for {r["id"]}'
        assert normalize(prompt) not in blocked, f'Held-out prompt overlap: {r["id"]}'
        if original['before_hash'] in seen or original['after_hash'] in seen:
            continue
        seen.update([original['before_hash'], original['after_hash']])
        for side in ['before','after']:
            with Image.open(original[side]) as im:
                assert im.size == (512,512)
        exported.append(dict(prompt_id=int(r['id']), sample_id=0, prompt=prompt,
                             before=original['before'], after=original['after'],
                             instruction=original['instruction'], tag=original['tag'],
                             approved=True, dataset=REPO, source_revision=source['revision'],
                             source_row=original['source_row'], notes=r['notes']))
    assert len(exported) >= a.target, f'Only {len(exported)} usable approved pairs; requested {a.target}. Review more or explicitly lower --target.'
    chosen = exported[:a.target]
    path = a.out/f'train_pairs_{a.target}.jsonl'
    assert not path.exists(), 'Export already exists; preserve the original training file'
    path.write_text(''.join(json.dumps(r)+'\n' for r in chosen))
    summary = dict(training_pairs=len(chosen), tags=dict(Counter(r['tag'] for r in chosen)),
                   prompt_texts=len({r['prompt'] for r in chosen}), path=str(path.resolve()),
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (a.out/f'train_summary_{a.target}.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare','review','export'])
    p.add_argument('--out', type=lambda s: Path(s).resolve(), required=True)
    p.add_argument('--candidates', type=int, default=3000)
    p.add_argument('--max-scan', type=int, default=20000)
    p.add_argument('--target', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-aspect', type=float, default=1.15)
    p.add_argument('--revision', default='main')
    p.add_argument('--model', default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    p.add_argument('--model-revision', default='451f4fe16113bff5a5d2269ed5ad43b0592e9a14')
    p.add_argument('--exclude', type=Path, nargs='+', default=[])
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare(a)
    elif a.mode == 'review':
        assert not (a.out/'review.csv').exists(), 'Will not overwrite an existing review'
        write_review(a.out, read_rows(a.out/'candidates.jsonl'))
    else:
        export(a)
