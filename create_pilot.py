"""Create a reviewed GenEval-compatible pilot using Python's standard library.
Shared objects require --mode combinations. Exact complete test specifications
are excluded; shared objects and individual facts are NOT excluded.
"""
import argparse
import csv
import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path

OBJECTS = dict(zip(
    'backpack,bottle,bowl,book,chair,cup,handbag,suitcase,umbrella,vase,tie,toothbrush,frisbee,skateboard,surfboard,clock,car,bicycle,motorcycle,bench'.split(','),
    'backpacks,bottles,bowls,books,chairs,cups,handbags,suitcases,umbrellas,vases,ties,toothbrushes,frisbees,skateboards,surfboards,clocks,cars,bicycles,motorcycles,benches'.split(',')))
COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'black', 'white']
RELATIONS = ['left of', 'right of', 'above', 'below']
WEIGHTS = {'counting': 20, 'color_binding': 20, 'spatial': 30,
           'color_spatial': 20, 'count_color': 10}


def signature(record):
    """Ignore wording/tag and normalize inverse relations and object order.
    Ignore exclude upper bounds conservatively: an exact-count candidate is
    excluded even when the matching test specification only has a lower bound.
    """
    items = record.get('include', [])
    nodes = [(o['class'], o['count'], o.get('color', '')) for o in items]
    edges = []
    for i, obj in enumerate(items):
        if 'position' in obj:
            relation, target = obj['position']
            a, b = nodes[i], nodes[target]
            if relation in ['right of', 'below']:
                a, b = b, a
                relation = {'right of': 'left of', 'below': 'above'}[relation]
            edges.append((a, relation, b))
    return tuple(sorted(nodes)), tuple(sorted(edges))


def make_record(names, counts, colors, category, relation=None):
    objects = []
    phrases = []
    for name, count, color in zip(names, counts, colors):
        obj = {'class': name, 'count': count}
        if color:
            obj['color'] = color
        objects.append(obj)
        noun = f"{color + ' ' if color else ''}{name if count == 1 else OBJECTS[name]}"
        # Vowel spelling works for this curated vocabulary and its colors.
        quantity = ('an' if noun[0] in 'aeiou' else 'a') if count == 1 else {2: 'two', 3: 'three'}[count]
        phrases.append(f'{quantity} {noun}')
    if relation:
        inverse = {
            'left of': 'right of',
            'right of': 'left of',
            'above': 'below',
            'below': 'above',
        }
        objects[1]['position'] = [inverse[relation], 0]
        wording = f'to the {relation}' if relation in ['left of', 'right of'] else relation
        prompt = f'{phrases[0]} {wording} {phrases[1]}.'
    else:
        prompt = ' and '.join(phrases) + '.'
    prompt = prompt[0].upper() + prompt[1:]
    tag = 'position' if relation else ('color_attr' if category == 'color_binding' else 'counting')
    return {'tag': tag, 'include': objects,
            'exclude': [{'class': o['class'], 'count': o['count'] + 1} for o in objects],
            'prompt': prompt}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geneval-repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['objects', 'combinations'], default='objects')
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--export-reviewed', action='store_true')
    args = p.parse_args()
    if args.n < 10:
        p.error('--n must be at least 10')
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = args.geneval_repo / 'prompts/evaluation_metadata.jsonl'
    test = [json.loads(s) for s in metadata.read_text().splitlines() if s.strip()]
    forbidden = {signature(r) for r in test}
    if args.export_reviewed:
        accepted = {'train': [], 'dev': []}
        seen, groups = set(), {}
        for row in csv.DictReader((args.out / 'review.csv').open()):
            approval = row['approved'].strip().lower()
            if approval in ['false', 'no', '0']:
                continue
            if approval not in ['', 'true', 'yes', '1']:
                raise ValueError(f"Unknown approval value: {row['approved']!r}")
            r = json.loads(row['metadata'])
            r['prompt'] = row['prompt']
            key = signature(r)
            if key in forbidden or key in seen:
                raise ValueError(f"Test overlap or duplicate in reviewed row {row['id']}")
            group = tuple(sorted(o['class'] for o in r['include']))
            if group in groups and groups[group] != row['split']:
                raise ValueError(f"Object-set split leakage in row {row['id']}")
            groups[group] = row['split']
            seen.add(key)
            accepted[row['split']].append(r)
        for split, rows in accepted.items():
            (args.out / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            print(f'{split}: {len(rows)} approved prompts')
        return
    if (args.out / 'review.csv').exists():
        raise SystemExit('review.csv already exists. Choose a new output directory to preserve your review.')
    vocabulary = set((args.geneval_repo / 'evaluation/object_names.txt').read_text().splitlines())
    test_objects = {o['class'] for r in test for k in ['include', 'exclude'] for o in r.get(k, [])}
    report = dict(test_prompts=len(test), detector_objects=len(vocabulary),
                  intersection=sorted(vocabulary & test_objects),
                  unused_detector_objects=sorted(vocabulary - test_objects),
                  curated_intersection=sorted(set(OBJECTS) & test_objects),
                  metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
                  mode=args.mode, seed=args.seed)
    (args.out / 'overlap_report.json').write_text(json.dumps(report, indent=2))
    print(f'Detector classes: {len(vocabulary)}; test overlap: {len(vocabulary & test_objects)}; unused: {len(vocabulary-test_objects)}')
    names = sorted(set(OBJECTS) & vocabulary)
    if args.mode == 'objects':
        names = [o for o in names if o not in test_objects]
    (args.out / 'objects.json').write_text(json.dumps({o: OBJECTS[o] for o in names}, indent=2))
    if len(names) < 2:
        raise SystemExit('No usable disjoint object pool. Use --mode combinations only if shared objects are acceptable.')
    pools = {category: {} for category in WEIGHTS}

    def add(category, objects, counts, colors, relation=None):
        row = make_record(objects, counts, colors, category, relation)
        key = signature(row)
        if key not in forbidden:
            pools[category][key] = row

    for pair in itertools.combinations(names, 2):
        # Two classes allow unseen counting specifications; many single-class
        # counts 1-3 are already present in the official test set.
        for counts in itertools.product([1, 2, 3], repeat=2):
            add('counting', pair, counts, ['', ''])
        for colors in itertools.permutations(COLORS, 2):
            add('color_binding', pair, [1, 1], colors)
            for relation in RELATIONS:
                add('color_spatial', pair, [1, 1], colors, relation)
        for relation in RELATIONS:
            add('spatial', pair, [1, 1], ['', ''], relation)
    for name, count, color in itertools.product(names, [1, 2, 3], COLORS):
        add('count_color', [name], [count], [color])
    rng = random.Random(args.seed)
    quotas = {k: args.n * w // 100 for k, w in WEIGHTS.items()}
    for category in list(WEIGHTS)[:args.n - sum(quotas.values())]:
        quotas[category] += 1
    rows = []
    for category, target in quotas.items():
        candidates = list(pools[category].values())
        if target > len(candidates):
            raise ValueError(f'{category}: requested {target}, only {len(candidates)} unique specifications available')
        rows += [(category, r) for r in rng.sample(candidates, target)]
    rng.shuffle(rows)
    groups = sorted({tuple(sorted(o['class'] for o in r['include'])) for _, r in rows})
    dev = set()
    for size in [1, 2]:
        subset = [g for g in groups if len(g) == size]
        rng.shuffle(subset)
        dev.update(subset[:max(1, round(len(subset) * .2))])
    splits = Counter()
    with (args.out / 'review.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'category', 'split', 'approved', 'notes', 'prompt', 'metadata'])
        writer.writeheader()
        for i, (category, r) in enumerate(rows):
            split = 'dev' if tuple(sorted(o['class'] for o in r['include'])) in dev else 'train'
            splits[split] += 1
            writer.writerow(dict(id=f'pilot_{i:05d}', category=category, split=split, approved='', notes='', prompt=r['prompt'], metadata=json.dumps(r)))
    (args.out / 'candidates.jsonl').write_text(''.join(json.dumps(r) + '\n' for _, r in rows))
    report.update(categories=quotas, splits=dict(splits), unique_specifications=len({signature(r) for _, r in rows}))
    (args.out / 'overlap_report.json').write_text(json.dumps(report, indent=2))
    print(f'Created {len(rows)} candidates: {quotas}; splits: {dict(splits)}')
    print('Review review.csv, set approved=yes, then run --export-reviewed.')
    print('Shared objects and atomic facts remain; matching complete test specifications are excluded.')


if __name__ == '__main__':
    main()
