"""Computer-vision model downloader.

  python -m app.download_models                 install the recommended models
  python -m app.download_models --list          show every model and its status
  python -m app.download_models --all           install everything in the registry
  python -m app.download_models NAME [NAME...]  install specific models
  python -m app.download_models --verify        re-check checksums of installed files
  python -m app.download_models NAME --force    re-download
  python -m app.download_models --add NAME URL [--sha256 HEX] [--filename F]
                                                register any extra model file and download it
  python -m app.download_models --dir D ...     use another model folder (also MODEL_DIR env)
"""
import argparse
import os
import sys


def _bar(label):
    state = {"last": -1}

    def cb(done, total):
        pct = int(done * 100 / total) if total else -1
        if total and pct == state["last"]:
            return
        state["last"] = pct
        txt = f"{pct:3d}%" if total else "   ?"
        print(f"\r  {label}: {done/1e6:7.1f} MB  {txt}", end="", flush=True)
    return cb


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m app.download_models", description="Download computer-vision models.")
    ap.add_argument("names", nargs="*", help="model names (default: recommended)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dir", help="model folder")
    ap.add_argument("--add", nargs=2, metavar=("NAME", "URL"))
    ap.add_argument("--sha256")
    ap.add_argument("--filename")
    a = ap.parse_args(argv)
    if a.dir:
        os.environ["MODEL_DIR"] = a.dir
    from . import model_registry as mr
    if a.dir:
        from pathlib import Path
        mr.MODEL_DIR = Path(a.dir)
        mr.CUSTOM_FILE = mr.MODEL_DIR / "custom_models.json"

    if a.add:
        try:
            mr.add_custom(a.add[0], a.add[1], a.filename, a.sha256)
        except ValueError as e:
            print("Error:", e); return 2
        a.names = [a.add[0]]
        print(f"Registered '{a.add[0]}'.")

    reg = mr.registry()
    if a.list or a.verify:
        print(f"Model folder: {mr.MODEL_DIR}\n")
        bad = 0
        for o in mr.overview():
            tag = "recommended" if o["recommended"] else ("custom" if not o["builtin"] else "optional")
            print(f"  {o['name']:<20} {o['state']:<10} {str(o['size_mb'] or '?'):>6} MB  [{tag}]  {o['title']}")
            if a.verify and o["state"] == "corrupt":
                bad += 1
        return 1 if bad else 0

    names = list(reg) if a.all else (a.names or [k for k, v in reg.items() if v.get("recommended")])
    unknown = [n for n in names if n not in reg]
    if unknown:
        print("Unknown model(s):", ", ".join(unknown), "\nRun with --list to see names."); return 2
    print(f"Model folder: {mr.MODEL_DIR}")
    fail = 0
    for n in names:
        if mr.is_installed(n) and not a.force:
            print(f"✓ {n}: already installed and verified"); continue
        print(f"↓ {n}: {reg[n]['title']}")
        try:
            mr.install(n, _bar(n), a.force)
            print(f"\r✓ {n}: installed and verified                    ")
        except Exception as e:      # noqa: BLE001
            fail += 1
            print(f"\n✗ {n}: {e}")
            if a.add and n == a.add[0]:           # don't keep a registration that never downloaded
                import json
                d = mr.load_custom(); d.pop(n, None)
                mr.CUSTOM_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
                print("  (registration removed)")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
