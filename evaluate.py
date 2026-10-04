"""Launch the official evaluator in its OWN environment, never reimplement it."""
import argparse
import subprocess
from pathlib import Path
from common import atomic_json, file_hash, read_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--geneval-repo', required=True)
    p.add_argument('--geneval-python', required=True, help='Absolute Python path in official GenEval environment')
    p.add_argument('--detector-dir', required=True)
    p.add_argument('--benchmark-source', default='data/geneval/source.json')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root = Path(a.run).resolve()
    repo = Path(a.geneval_repo).resolve()
    source = read_json(a.benchmark_source)
    manifest = read_json(root / 'manifest.json')
    if manifest['spec']['metadata_sha256'] != source['sha256']:
        raise ValueError('Run input does not match the pinned benchmark source.')
    audit = read_json(root / 'audit.json')
    if not audit['complete'] or audit['scope'] == 'smoke':
        raise ValueError('Run a successful full-input audit before evaluation.')
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != source['revision']:
        raise ValueError('Evaluator checkout must match the benchmark source revision.')
    if subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD', '--', 'evaluation'], text=True).strip():
        raise ValueError('Official evaluator has local modifications.')
    out = Path(a.out).resolve()
    if out.exists():
        raise ValueError('Choose a fresh results filename; do not overwrite evaluation evidence.')
    out.parent.mkdir(parents=True, exist_ok=True)
    python = str(Path(a.geneval_python).resolve())
    command = [python, str(repo/'evaluation/evaluate_images.py'), str(root), '--outfile', str(out),
               '--model-path', str(Path(a.detector_dir).resolve())]
    subprocess.run(command, cwd=repo, check=True)
    packages = subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True)
    # Weights can be large; checksum once per evaluation to record exact evaluator artifacts.
    weights = {str(p.relative_to(a.detector_dir)): file_hash(p) for p in Path(a.detector_dir).rglob('*') if p.is_file()}
    atomic_json(out.with_suffix('.provenance.json'), {'command': command, 'geneval_commit': commit,
                'detector_files_sha256': weights, 'pip_freeze': packages,
                'results_sha256': file_hash(out), 'manifest_sha256': file_hash(root/'manifest.json')})

if __name__ == '__main__':
    main()
