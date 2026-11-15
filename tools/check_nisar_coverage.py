"""Перевірка покриття NISAR GCOV для AOI через публічний каталог NASA CMR (логін не потрібен).

Приклад:
    python -I -X utf8 Project/tools/check_nisar_coverage.py --name lake_chad --bbox 13.0,12.3,15.5,14.5

Пише у Project/data_plan/<name>_coverage.md таблицю "трек/кадр -> дати" та
<name>_urls_<track>_<dir>_<frame>.txt зі списком HDF5 для найкращих стеків.
Завантаження самих файлів потребує Earthdata Login.
"""
import argparse
import collections
import json
import pathlib
import urllib.parse
import urllib.request

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json?"
COLLECTIONS = {
    "PROVISIONAL": "C2854338529-ASF",  # зйомки 17.06.2026 .. ~21.10.2026
    "VALIDATED": "C3622241360-ASF",    # зйомки від ~21.10.2026
    "BETA": "C3622214170-ASF",         # некалібровані, 10.2025 .. 01.2026: у стек не змішувати
}


def fetch(concept_id, bbox):
    out, page = [], 1
    while True:
        q = urllib.parse.urlencode({
            "collection_concept_id": concept_id, "bounding_box": bbox,
            "page_size": 2000, "page_num": page, "sort_key": "start_date"})
        req = urllib.request.Request(CMR + q, headers={"User-Agent": "python-urllib"})
        entries = json.load(urllib.request.urlopen(req, timeout=90))["feed"]["entry"]
        out += entries
        if len(entries) < 2000:
            return out
        page += 1


def extent(entry):
    """Межі кадру (W, S, E, N) з полігона CMR (пари lat lon)."""
    nums = [float(v) for v in entry.get("polygons", [[""]])[0][0].split()]
    lats, lons = nums[0::2], nums[1::2]
    return (min(lons), min(lats), max(lons), max(lats)) if nums else None


def overlap_share(ext, bbox):
    """Частка AOI, яку накриває прямокутна оболонка кадру (грубо, кадр нахилений)."""
    w, s, e, n = bbox
    if not ext:
        return 0.0
    iw, ie = max(w, ext[0]), min(e, ext[2])
    i_s, i_n = max(s, ext[1]), min(n, ext[3])
    if ie <= iw or i_n <= i_s:
        return 0.0
    return (ie - iw) * (i_n - i_s) / ((e - w) * (n - s))


def h5_url(entry):
    for link in entry.get("links", []):
        href = link.get("href", "")
        if href.startswith("https") and href.endswith(".h5"):
            return href
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--bbox", required=True, help="W,S,E,N у градусах")
    ap.add_argument("--top", type=int, default=4, help="для скількох стеків писати списки URL")
    args = ap.parse_args()
    bbox = tuple(float(v) for v in args.bbox.split(","))
    out_dir = pathlib.Path(__file__).resolve().parents[1] / "data_plan"
    out_dir.mkdir(parents=True, exist_ok=True)

    stacks = collections.defaultdict(list)
    for level, cid in COLLECTIONS.items():
        for e in fetch(cid, args.bbox):
            t = e["title"].split("_")
            # NISAR_L2_PR_GCOV_<cycle>_<track>_<dir>_<frame>_<mode>_<pol>_..._<start>_<end>_<CRID>_...
            key = (t[5], t[6], t[7], t[8], t[9])
            stacks[key].append({
                "level": level, "date": e["time_start"][:10], "crid": t[13] if len(t) > 13 else "",
                "mb": float(e.get("granule_size") or 0), "url": h5_url(e), "ext": extent(e), "title": e["title"]})

    def calibrated_dates(items):
        return sorted({i["date"] for i in items if i["level"] != "BETA"})

    ranked = sorted(stacks.items(), key=lambda kv: -len(calibrated_dates(kv[1])))
    lines = [f"# NISAR GCOV покриття: {args.name}", "",
             f"AOI (W,S,E,N): `{args.bbox}`. Джерело: NASA CMR, запит без логіну. "
             "Стек для статистики будувати з одного рядка (один трек, напрямок, кадр, режим).", "",
             "| Трек | Напр. | Кадр | Режим | Пол. | PROV дат | VALID дат | BETA дат | Перша .. остання (калібр.) | Оболонка кадру накриває AOI | GB на стек |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, items in ranked:
        n = {lv: len({i["date"] for i in items if i["level"] == lv}) for lv in COLLECTIONS}
        cal = calibrated_dates(items)
        if not cal:
            continue
        share = max(overlap_share(i["ext"], bbox) for i in items)
        gb = sum(i["mb"] for i in items if i["level"] != "BETA") / 1024
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | {n['PROVISIONAL']} | {n['VALIDATED']} | "
                     f"{n['BETA']} | {cal[0]} .. {cal[-1]} | {share:.0%} | {gb:.1f} |")
    for key, items in ranked[: args.top]:
        cal = sorted((i for i in items if i["level"] != "BETA"), key=lambda i: i["date"])
        if not cal:
            continue
        path = out_dir / f"{args.name}_urls_{key[0]}_{key[1]}_{key[2]}.txt"
        path.write_text("\n".join(f"{i['date']}\t{i['level']}\t{i['crid']}\t{i['mb']:.0f}MB\t{i['url']}" for i in cal) + "\n",
                        encoding="utf-8")
    md = out_dir / f"{args.name}_coverage.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nЗаписано: {md}")


if __name__ == "__main__":
    main()
