"""Схема пайплайна (та сама, що в документах) як PNG високої роздільності, EN і UA.

    python -I tools/pipeline_diagram.py
"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

PROJECT = Path(__file__).resolve().parents[1]
EDGE, ACC, INK, QUIET, TINT = '#c3c2bb', '#2a78d6', '#1f1f1d', '#6b6a63', '#efeeea'
W, H = 160, 72
TEXT = {
    'en': dict(title='From raw NISAR files to a change map: seven steps, each with its own check',
               boxes=[('NISAR L2 GCOV', '22 scenes, 2 tracks', 'HH and HV, Jun–Nov'), ('Area stack', 'one UTM grid, 80 m', 'window from each HDF5'),
                      ('RFI background R', 'dark land pixels', '+ permanent water'), ('Cleaning', 'Y − R, calibrated', 'on held-out water'),
                      ('Change test', 'Gamma(n, s + R)', 'FDR 5 %, ENL 29.8'), ('Validation', 'two tracks, per pixel', 'bed vs DSWx-S1'),
                      ('Web app', 'before/after, changes', 'two-track agreement')],
               runs=('raw files to website', 'in four commands'), water='permanent water',
               refs='References: OPERA DSWx-S1 · JRC Global Surface Water · ESA WorldCover · second track'),
    'uk': dict(title='Від сирих файлів NISAR до карти змін: сім кроків, кожен зі своєю перевіркою',
               boxes=[('NISAR L2 GCOV', '22 сцени, 2 треки', 'HH і HV, черв–лист'), ('Стек району', 'одна сітка UTM, 80 м', 'вікно з кожного HDF5'),
                      ('Фон завади R', 'темні пікселі суші', '+ постійна вода'), ('Очищення', 'Y − R, калібрування', 'водою в фолдах'),
                      ('Тест змін', 'Gamma(n, s + R)', 'FDR 5 %, ENL 29.8'), ('Перевірки', 'два треки попіксельно', 'дно проти DSWx-S1'),
                      ('Вебзастосунок', 'до / після, карти змін', 'збіг двох треків')],
               runs=('від сирих файлів до сайту', 'чотирма командами'), water='постійна вода',
               refs='Еталони: OPERA DSWx-S1 · JRC Global Surface Water · ESA WorldCover · другий трек'),
}
POS = [(24, 74), (208, 74), (392, 74), (576, 74), (24, 214), (208, 214), (392, 214)]


def svg(t):
    o = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 760 392' width='760' height='392' font-family='Segoe UI, Inter, sans-serif' font-size='13'>",
         "<rect width='760' height='392' fill='white'/>",
         f"<defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' orient='auto-start-reverse'><path d='M0 0L10 5L0 10z' fill='{EDGE}'/></marker></defs>",
         f"<text x='24' y='34' font-size='15' font-weight='600' fill='{INK}'>{t['title']}</text>",
         f"<g fill='none' stroke='{EDGE}' stroke-width='1.25'>"]
    for d in ('M184 110H206', 'M368 110H390', 'M552 110H574', 'M656 146V180H104V212', 'M184 250H206', 'M368 250H390', 'M288 326V288'):
        o.append(f"<path d='{d}' marker-end='url(#a)'/>")
    o.append("<path d='M564 326V160H500V148' stroke-dasharray='4 4' marker-end='url(#a)'/></g>")
    for k, ((x, y), (name, l1, l2)) in enumerate(zip(POS, t['boxes'])):
        main = k == 4
        o.append(f"<rect x='{x}' y='{y}' width='{W}' height='{H}' rx='8' fill='{ACC if main else 'none'}' fill-opacity='{0.12 if main else 1}' "
                 f"stroke='{ACC if main else EDGE}' stroke-width='{2 if main else 1.25}'/>")
        c = INK if main else QUIET
        o += [f"<text x='{x + 12}' y='{y + 24}' font-weight='600' fill='{INK}'>{name}</text>",
              f"<text x='{x + 12}' y='{y + 44}' font-size='11.5' fill='{c}'>{l1}</text>",
              f"<text x='{x + 12}' y='{y + 60}' font-size='11.5' fill='{c}'>{l2}</text>"]
    o += [f"<text x='576' y='246' font-size='11.5' fill='{QUIET}'>{t['runs'][0]}</text>",
          f"<text x='576' y='262' font-size='11.5' fill='{QUIET}'>{t['runs'][1]}</text>",
          f"<rect x='24' y='326' width='712' height='42' rx='8' fill='{TINT}' stroke='{EDGE}' stroke-width='1.25'/>",
          f"<text x='380' y='352' text-anchor='middle' fill='{INK}'>{t['refs']}</text>",
          f"<text x='572' y='312' font-size='11.5' fill='{QUIET}'>{t['water']}</text></svg>"]
    return ''.join(o)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    out = {'en': PROJECT / 'Report figures (EN)' / '11_pipeline_diagram.png',
           'uk': PROJECT / 'Ілюстрації доповіді' / '11_схема_пайплайна.png'}
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={'width': 760, 'height': 392}, device_scale_factor=3)
        for lang, path in out.items():
            pg.set_content(f"<html><body style='margin:0'>{svg(TEXT[lang])}</body></html>")
            pg.locator('svg').screenshot(path=str(path))
            print(path)
        b.close()


if __name__ == '__main__':
    main()
