"""Review GenEval repairs, then export approved images for Diffusers LoRA training."""
import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path


def load(path):
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        p = Path(row['filename'])
        key = (int(p.parent.parent.name), int(p.stem))
        assert key not in result, f'Duplicate image: {key}'
        assert isinstance(row['correct'], bool), f'Invalid correctness: {key}'
        result[key] = row
    return result


def metadata(row):
    value = row.get('metadata')
    return json.loads(value) if isinstance(value, str) else value


def prepare(args):
    import base64
    import io
    from PIL import Image

    before, after = load(args.before), load(args.after)
    assert before.keys() == after.keys(), 'Expected evaluation of the full same split'
    args.out.mkdir(parents=True, exist_ok=True)
    review_path = args.out / 'review.csv'
    assert not review_path.exists(), 'Review exists; use a new output folder to avoid losing labels'
    rows, cards = [], []
    for key, old in before.items():
        new = after[key]
        assert metadata(old) == metadata(new), f'Metadata mismatch: {key}'
        if old['correct'] == new['correct']:
            continue
        pid, sid = key
        kind = 'repair' if new['correct'] else 'regression'
        prompt = old.get('prompt') or metadata(old)['prompt']
        row = dict(prompt_id=pid, sample_id=sid, kind=kind,
                   tag=old['tag'], prompt=prompt,
                   before=str(Path(old['filename']).absolute()),
                   after=str(Path(new['filename']).absolute()),
                   before_reason=old.get('reason', ''), after_reason=new.get('reason', ''),
                   approved='', notes='')
        for field in ('before', 'after'):
            assert Path(row[field]).is_file(), row[field]
        rows.append(row)
        pictures = []
        for field in ('before', 'after'):
            with Image.open(row[field]) as im:
                im = im.convert('RGB')
                im.thumbnail((512, 512))
                buf = io.BytesIO()
                im.save(buf, format='JPEG', quality=90)
            encoded = base64.b64encode(buf.getvalue()).decode()
            pictures.append(f'<figure><figcaption>{field}</figcaption><img src="data:image/jpeg;base64,{encoded}"></figure>')
        title = f'{pid:05d}/{sid:04d} — {kind} — {row["tag"]}'
        cards.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(prompt)}</p>'
                     + '<div>' + ''.join(pictures) + '</div><p>Before: '
                     + html.escape(str(row['before_reason'])) + '</p></section>')
    assert rows, 'No changed correctness outcomes found'
    with review_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / 'review.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>DFAR repair audit</title><style>body{font:16px sans-serif;margin:32px}'
        'section{border-bottom:1px solid #ccc;padding:20px 0}div{display:flex;flex-wrap:wrap}'
        'figure{margin:8px}img{max-width:100%}</style><h1>DFAR repair audit</h1>'
        '<p>Label review.csv. TRUE means the corrected image satisfies the entire prompt, '
        'visibly repairs the error, and has acceptable image quality. Blank means unreviewed. '
        'Regression rows are diagnostic only.</p>' + ''.join(cards))
    print(dict(Counter(row['kind'] for row in rows)))
    print('Open:', args.out / 'review.html', '\nLabel:', review_path)


def export(args):
    from PIL import Image

    with (args.out / 'review.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    accepted = [r for r in rows if r['kind'] == 'repair'
                and r['approved'].strip().lower() == 'true']
    assert accepted, 'No approved repairs; label TRUE only after visual review'
    dataset = args.out / 'train_images'
    dataset.mkdir(exist_ok=False)  # Do not silently retain images from an older export.
    records = []
    for r in accepted:
        source = Path(r['after']).resolve(strict=True)
        with Image.open(source) as im:
            assert im.size == (512, 512), f'Expected 512x512, got {im.size}: {source}'
        name = f'{int(r["prompt_id"]):05d}_{int(r["sample_id"]):04d}.png'
        (dataset / name).symlink_to(source)
        records.append(json.dumps({'file_name': name, 'text': r['prompt']}))
    (dataset / 'metadata.jsonl').write_text('\n'.join(records) + '\n')
    (args.out / 'approved_pairs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in accepted))
    summary = {'approved_images': len(accepted),
               'unique_prompt_ids': len({r['prompt_id'] for r in accepted}),
               'unique_prompt_texts': len({r['prompt'] for r in accepted}),
               'per_tag': dict(Counter(r['tag'] for r in accepted))}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('Training images:', dataset)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'export'])
    p.add_argument('--before', type=Path)
    p.add_argument('--after', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.mode == 'prepare':
        assert args.before and args.after, '--before and --after are required'
        prepare(args)
    else:
        export(args)
