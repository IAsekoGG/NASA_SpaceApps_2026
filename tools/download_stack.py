"""Довантажує повні .h5 NISAR GCOV зі списку URL (data_plan/*_urls_*.txt) у Project/data/h5, пропускаючи наявні.

    python -X utf8 Project/tools/download_stack.py --urls Project/data_plan/kakhovka_urls_143_A_025.txt
    python -X utf8 Project/tools/download_stack.py --urls ... --levels PROVISIONAL,VALIDATED --threads 3

Потрібен Earthdata Login у ~/_netrc (або ~/.netrc).
"""
import argparse
import pathlib
import shutil

PROJECT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True)
    ap.add_argument("--dest", default=str(PROJECT / "data" / "h5"))
    ap.add_argument("--levels", default="PROVISIONAL,VALIDATED")
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()

    dest = pathlib.Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    levels = set(args.levels.split(","))
    rows = [l.split("\t") for l in pathlib.Path(args.urls).read_text(encoding="utf-8").splitlines() if l.count("\t") == 4]
    todo, need_mb = [], 0
    for date, level, _, size, url in rows:
        if level not in levels:
            continue
        name = url.rsplit("/", 1)[-1]
        if (dest / name).exists():
            print("є:", date, name[:60])
            continue
        todo.append(url)
        need_mb += int(size.rstrip("MB"))
    free_mb = shutil.disk_usage(dest).free / 2**20
    print(f"Качати {len(todo)} файлів, {need_mb / 1024:.1f} ГБ; вільно {free_mb / 1024:.0f} ГБ")
    if not todo:
        return
    if need_mb > free_mb - 5 * 1024:
        raise SystemExit("Замало місця на диску (лишаю запас 5 ГБ).")

    import earthaccess
    if not earthaccess.login(strategy="netrc").authenticated:
        raise SystemExit("Немає Earthdata Login у _netrc")
    earthaccess.download(todo, str(dest), threads=args.threads)
    for url in todo:
        p = dest / url.rsplit("/", 1)[-1]
        print("OK " if p.exists() else "НЕМАЄ ", p.name, f"{p.stat().st_size / 2**20:.0f} MB" if p.exists() else "")


if __name__ == "__main__":
    main()
