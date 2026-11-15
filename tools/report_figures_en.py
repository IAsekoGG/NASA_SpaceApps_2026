"""Англійський набір ілюстрацій доповіді → Project/report_figures_en/.

Ті самі скрипти, що й для українського набору; підписи перекладає tools/figtext.py.

    python -I tools/report_figures_en.py
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'tools'))
import figtext  # noqa: E402
figtext.install()

import matplotlib.pyplot as plt  # noqa: E402
import report_figures as A  # noqa: E402
LIGHT = dict(plt.rcParams)          # світлий стиль report_figures.py; темні скрипти нижче перезаписують rcParams
import report_figures_wow as B  # noqa: E402
import report_figures_dualpol as C  # noqa: E402
import report_cover as V  # noqa: E402

OUT = PROJECT / 'report_figures_en'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    OUT.mkdir(exist_ok=True)
    for m in (A, B, C, V):
        m.OUT = OUT
    dark = dict(plt.rcParams)
    with plt.rc_context(LIGHT):
        for f in (A.fig_s2_story, A.fig_blinded, A.fig_before_after, A.fig_change_map, A.fig_agreement):
            f(); print('done:', f.__name__)
    plt.rcParams.update(dark)
    rgb = B.fig_radar_rgb(); B.fig_diagonal_split(); B.fig_timelapse(); B.fig_concept(); B.fig_radar_vs_optics(rgb); print('done: wow')
    C.fig_compare('108_D_065', '2026-07-03', C.REF['108_D_065'], '108D', '108 D (вечірній)')
    C.fig_compare('143_A_025', '2026-07-18', C.REF['143_A_025'], '143A', '143 A (нічний)')
    C.fig_season_grid('143_A_025', '143 A'); C.fig_density('143_A_025', '2026-07-18', C.REF['143_A_025'], '143 A')
    C.gif_dualpol('143_A_025', '143 A'); print('done: dual-pol')
    sys.argv = ['report_cover.py', '--date', '2026-08-23', '--c', '1.10']
    V.main(); print('done: cover')
    print('Cyrillic left:', sorted(figtext.LEFTOVER) or 'none')


if __name__ == '__main__':
    main()
