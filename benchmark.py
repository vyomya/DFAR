"""Run manual, generic, or Qwen-guided VAREdit experiments on existing images.

Uses the existing locally patched VAREdit-8B-512 loader (two GPUs).
No training, image regeneration, or evaluation is performed here.
Use a fresh --out directory for each run; this minimal script does not resume.
"""
import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_RUN = Path("/fs/nexus-scratch/vyomwal5/dfar_results/runs/baseline/sd15")
CFG, TAU, SEED = 3.0, 0.1, 42


def read_metadata(folder):
    text = (folder / "metadata.jsonl").read_text().strip()
    # Support one JSON object, JSONL, and older Python-dict metadata.
    for candidate in (text, text.splitlines()[0] if text else ""):
        try:
            metadata = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                metadata = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
        if isinstance(metadata, dict) and "prompt" in metadata:
            return metadata
    raise ValueError(f"Invalid metadata: {folder / 'metadata.jsonl'}")


def make_cases(mode, annotations, originals):
    """Only select source IDs; original prompts always come from run metadata."""
    if mode == "generic":
        return [
            dict(prompt_id=pid, sample_id=sid, decision="edit", instruction=(
                f'The original generation prompt was: "{read_metadata(image.parent.parent)["prompt"]}". '
                "Edit this image so it satisfies that prompt. "
                "Preserve content that already matches the prompt. "
                "If the image already satisfies the prompt, leave it unchanged."
            ))
            for (pid, sid), image in originals.items()
        ]
    selected = {(int(c["prompt_id"]), int(c["sample_id"])) for c in annotations}
    if len(selected) != len(annotations):
        raise ValueError("Duplicate prompt/sample IDs in cases file")
    if not selected <= originals.keys():
        raise ValueError(f"Selected images missing: {sorted(selected - originals.keys())}")
    if mode == "qwen" and selected != originals.keys():
        raise ValueError("Qwen JSON must contain a decision for every original image")
    cases = []
    for c in annotations:
        pid, sid = int(c["prompt_id"]), int(c["sample_id"])
        prompt = read_metadata(originals[(pid, sid)].parent.parent)["prompt"]
        if c.get("original_prompt") and c["original_prompt"].strip() != prompt.strip():
            raise ValueError(f"Original prompt mismatch: {pid}/{sid}")
        if mode == "manual":
            decision = "edit"
            instruction = (c.get("instruction") or "").strip()
            if not instruction and c.get("is_control"):
                instruction = "Keep this image unchanged. Preserve all objects, colors, positions, and background."
        else:
            decision = c["decision"]
            if decision not in {"edit", "keep", "review"}:
                raise ValueError(f"Invalid Qwen decision: {decision}")
            instruction = c.get("instruction") if decision == "edit" else None
            if decision == "edit" and c.get("status", "ok") != "ok":
                raise ValueError(f"Invalid Qwen output marked for editing: {pid}/{sid}")
        if decision == "edit" and (not isinstance(instruction, str) or not instruction.strip()):
            raise ValueError(f"Missing editing instruction: {pid}/{sid}")
        cases.append(dict(prompt_id=pid, sample_id=sid, decision=decision,
                          instruction=instruction, status=c.get("status"),
                          is_control=c.get("is_control", False)))
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["manual", "generic", "qwen"], required=True)
    p.add_argument("--cases", type=Path, help="Manual annotations or complete Qwen JSON")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run", type=Path, default=SOURCE_RUN)
    p.add_argument("--varedit-repo", type=Path, default=ROOT / "external/VAREdit")
    a = p.parse_args()
    source_run, out, repo = a.run.resolve(), a.out.resolve(), a.varedit_repo.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Choose a fresh output directory; no resume implemented: {out}")
    originals = {}
    for image in sorted(source_run.glob("*/samples/*.png")):
        if image.parent.parent.name.isdigit() and image.stem.isdigit():
            key = (int(image.parent.parent.name), int(image.stem))
            if key in originals:
                raise ValueError(f"Duplicate source image ID: {key}")
            originals[key] = image
    if not originals:
        raise ValueError(f"No source images found: {source_run}")
    if a.mode != "generic" and a.cases is None:
        p.error("--cases is required for manual and qwen modes")
    annotations = json.loads(a.cases.read_text()) if a.cases else []
    cases = make_cases(a.mode, annotations, originals)
    if not cases:
        raise ValueError("No images selected")

    # Respect exports from the terminal. Defaults retain the existing scratch cache.
    os.environ.setdefault("HF_HOME", "/fs/nexus-scratch/vyomwal5/.cache/huggingface")
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    components, weights, commit = None, None, None
    if any(c["decision"] == "edit" for c in cases):
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        import torch
        from huggingface_hub import snapshot_download
        from infer import load_model, generate_image
        if torch.cuda.device_count() < 2:
            raise RuntimeError("The existing patched VAREdit loader requires two visible GPUs.")
        torch.cuda.set_device(0)
        weights = Path(snapshot_download(
            "HiDream-ai/VAREdit",
            allow_patterns=["8B-512.pth", "infinity_vae_d56_f8_14_patchify.pth", "flan-t5-xl/*"],
        ))
        print("Loading VAREdit on", torch.cuda.get_device_name(0), flush=True)
        components = load_model(str(weights), str(weights / "8B-512.pth"), "8B", 512)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    out.mkdir(parents=True, exist_ok=True)
    settings = dict(mode=a.mode, source_run=str(source_run), model="HiDream-ai/VAREdit",
                    checkpoint="8B-512.pth", cfg=CFG, tau=TAU, seed=SEED,
                    weights_snapshot=weights.name if weights else None, code_commit=commit)
    if components is not None:
        settings.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    (out / "cases.json").write_text(json.dumps(cases, indent=2))
    print(f"Mode: {a.mode} | Images: {len(cases)} | Output: {out}", flush=True)

    for case in cases:
        pid, sid = case["prompt_id"], case["sample_id"]
        source = originals[(pid, sid)]
        metadata = read_metadata(source.parent.parent)
        target = out / "after" / f"{pid:05d}"
        (target / "samples").mkdir(parents=True, exist_ok=True)
        output_image = target / "samples" / f"{sid:04d}.png"
        (target / "metadata.jsonl").write_text(json.dumps(metadata) + "\n")
        if case["decision"] != "edit":
            # No copied original: symlink ensures keep/review still get evaluated.
            output_image.symlink_to(source)
            seconds, action = 0.0, "retained_original"
        else:
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                edited = generate_image(components, str(source), case["instruction"],
                                        cfg=CFG, tau=TAU, seed=SEED)
            torch.cuda.synchronize()
            seconds, action = time.perf_counter() - start, "edited"
            edited.save(output_image)
        record = dict(**case, action=action, source=str(source),
                      prompt=metadata["prompt"], seconds=seconds)
        (out / f"{pid:05d}_{sid:04d}.json").write_text(json.dumps(record, indent=2))
        print(f"{pid}/{sid}: {action}, {seconds:.2f}s", flush=True)
    print(f"Complete. Evaluate: {out / 'after'}")


if __name__ == "__main__":
    main()
