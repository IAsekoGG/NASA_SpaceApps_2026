"""Англійські підписи для рисунків доповіді без зміни самих скриптів.

install() перехоплює кожен текст, що потрапляє на рисунок (matplotlib Text і PIL ImageDraw.text),
і перекладає його за таблицею фраз. LEFTOVER збирає підписи, де лишилась кирилиця.
"""
import re

PHRASES = {
    # report_figures.py
    'Каховське водосховище за три роки: від моря до лісу': 'Kakhovka Reservoir in three years: from a sea to a forest',
    'Sentinel-2, справжні кольори, 60 м · той самий район 140 × 100 км, що й у радарному аналізі':
        'Sentinel-2 true colour, 60 m · the same 140 × 100 km area as in the radar analysis',
    'Містить модифіковані дані Copernicus Sentinel (2023–2026), обробка команди.':
        'Contains modified Copernicus Sentinel data (2023–2026), processed by the team.',
    'травень 2023 · водосховище': 'May 2023 · reservoir',
    'червень 2023 · після руйнування греблі': 'June 2023 · after the dam breach',
    'літо 2024 · дно заростає': 'summer 2024 · the bed regrows',
    'літо 2026 · молодий ліс на дні': 'summer 2026 · young forest on the bed',
    'збільшено: поля видно крізь пляму': 'zoom: fields visible through the blob',
    'Радар «засліплений»: перешкода лягає поверх знімка, але не стирає його':
        'A “blinded” radar: interference lies on top of the image but does not erase it',
    'NISAR L2 GCOV, трек 143 A, HH frequencyB 160 м, шкала −22…−2 дБ': 'NISAR L2 GCOV, track 143 A, HH frequency B, 160 m, scale −22…−2 dB',
    'Дані: NASA/JPL, ISRO — NISAR L2 GCOV (PROVISIONAL). Розмиті світлі плями — радіозавади; їхні джерела не аналізуються.':
        'Data: NASA/JPL, ISRO — NISAR L2 GCOV (PROVISIONAL). Diffuse bright blobs are radio-frequency interference; their sources are not analysed.',
    'До і після: поля повертаються з-під радіотуману': 'Before and after: fields emerge from the radio fog',
    'NISAR, трек 143 A, 06.07.2026, HH frequencyB 160 м, однакова шкала −22…−2 дБ для всіх панелей':
        'NISAR, track 143 A, 6 July 2026, HH frequency B, 160 m, same −22…−2 dB scale for all panels',
    'Дані: NASA/JPL, ISRO — NISAR L2 GCOV. Очищення: адитивна модель Y = s + R, фон оцінено по темних пікселях суші й постійній воді.':
        'Data: NASA/JPL, ISRO — NISAR L2 GCOV. Cleaning: additive model Y = s + R, background estimated from dark land pixels and permanent water.',
    'коли почався новий стан поверхні (середина проміжку між знімками)': 'when the new surface state began (midpoint between acquisitions)',
    'Карта змін: коли на кожному полі сталася перша зміна': 'Change map: when each field first changed',
    'Тест змін із відомим фоном перешкоди, усі 11 дат, 160 м, FDR 5 % · тло — медіана очищеного HH':
        'Interference-aware change test, all 11 dates, 160 m, FDR 5 % · background: median cleaned HH',
    'Дані: NASA/JPL, ISRO — NISAR L2 GCOV. Липень — збирання зернових; серпень–вересень — соняшник, оранка; жовтень — кукурудза, сівба.':
        'Data: NASA/JPL, ISRO — NISAR L2 GCOV. July: cereal harvest; August–September: sunflower, ploughing; October: maize, sowing.',
    'дата першої зміни (середина проміжку між знімками)': 'date of first change (midpoint between acquisitions)',
    'Дані: NASA/JPL, ISRO — NISAR L2 GCOV. Треки летять у різний час доби з різних боків і обробляються незалежно.':
        'Data: NASA/JPL, ISRO — NISAR L2 GCOV. The tracks fly at different times of day from opposite sides and are processed independently.',
    'сильна перешкода': 'strong interference', 'чиста дата': 'clean date', ' · рілля HH ': ' · cropland HH ',
    'збільшено 24 × 16 км': 'zoom 24 × 16 km', 'збільшено 24 × 18 км: цілі поля, одна дата': 'zoom 24 × 18 km: whole fields, one date',
    'збіг: обидва треки, ±12 діб': 'agreement: both tracks, ±12 days',
    'Два незалежні треки бачать ті самі поля: у ': 'Two independent tracks see the same fields ',
    ' раза частіше за випадковість': '× more often than chance',
    'Одне вікно 26 × 18 км, три погляди · дати першої зміни на ріллі збігаються в ':
        'One 26 × 18 km window, three views · first-change dates on cropland agree for ',
    ' % пікселів проти ': ' % of pixels vs ', ' % випадково': ' % by chance',
    'до: сирий знімок': 'before: raw image', 'після: оцінку перешкоди віднято': 'after: interference estimate removed',
    'трек 143 A · змінилось ': 'track 143 A · changed ',
    # report_figures_wow.py
    'Сезон очима радара: кожне поле має свій колір': 'The season through radar eyes: every field has its own colour',
    'Очищений HH NISAR, трек 143 A · канал R — медіана червня–липня, G — серпня–вересня, B — жовтня–листопада\nсірий і білий — поверхня стабільна, кольори — поле змінилось':
        'Cleaned NISAR HH, track 143 A · R: June–July median, G: August–September, B: October–November\ngrey and white: stable surface, colour: the field changed',
    'Дані: NASA/JPL, ISRO — NISAR L2 GCOV, 11 дат, 160 м. Червоні поля: врожай зібрано після липня; сині: зійшли озимі або зорано восени.':
        'Data: NASA/JPL, ISRO — NISAR L2 GCOV, 11 dates, 160 m. Red fields: harvested after July; blue: winter crops emerged or autumn ploughing.',
    'СИРИЙ ЗНІМОК': 'RAW IMAGE', 'ПІСЛЯ ОЧИЩЕННЯ': 'AFTER CLEANING', 'СИРІ ЗНІМКИ': 'RAW IMAGES',
    'Один знімок, дві реальності': 'One image, two realities',
    'NISAR, трек 143 A, 06.07.2026, HH 160 м · ліворуч унизу те, що надіслав супутник, праворуч угорі — після нашого пайплайна; лінія ріже пляму навпіл':
        'NISAR, track 143 A, 6 July 2026, HH 160 m · bottom left: what the satellite sent; top right: after our pipeline; the line cuts the blob in half',
    'Дані: NASA/JPL, ISRO — NISAR L2 GCOV. Однакова шкала −22…−2 дБ для обох частин.':
        'Data: NASA/JPL, ISRO — NISAR L2 GCOV. Same −22…−2 dB scale for both halves.',
    'Ідея методу в одному рядку': 'The method in one line',
    'Синтетична ілюстрація: поля, спекл і пляма згенеровані, а не взяті з реального знімка.':
        'Synthetic illustration: the fields, speckle and blob are simulated, not taken from a real image.',
    'Оптика бачить колір, радар бачить зміну': 'Optics sees colour, radar sees change',
    'Той самий район 140 × 100 км · радар працює вночі й крізь хмари, а сезонний RGB показує, коли кожне поле змінилось':
        'The same 140 × 100 km area · radar works at night and through clouds, and the seasonal RGB shows when each field changed',
    'Дані: Copernicus Sentinel-2 (модифіковані); NASA/JPL, ISRO — NISAR L2 GCOV.': 'Data: Copernicus Sentinel-2 (modified); NASA/JPL, ISRO — NISAR L2 GCOV.',
    'NISAR, трек 143 A, HH · кожен кадр — новий проліт супутника': 'NISAR, track 143 A, HH · every frame is a new satellite pass',
    'поверхня s': 'surface s', 'поля різної яскравості': 'fields of different brightness', 'завада R': 'interference R',
    'гладка пляма, додається': 'a smooth blob, added on top', 'знімок Y = s + R': 'image Y = s + R', 'що бачить супутник': 'what the satellite sees',
    'очищення: Y − оцінка R': 'cleaning: Y − estimated R', 'поля повернулись': 'the fields are back',
    'червень–липень': 'June–July', 'серпень–вересень': 'August–September', 'жовтень–листопад': 'October–November',
    'оптика: Sentinel-2, літо 2026': 'optical: Sentinel-2, summer 2026', 'радар: NISAR, сезонний RGB': 'radar: NISAR, seasonal RGB',
    # report_figures_dualpol.py, report_cover.py
    'Перешкода міняє колір, а не лише яскравість': 'Interference changes colour, not just brightness',
    'Зелені плями мандрують від дати до дати — поля залишаються на місці': 'Green blobs wander from date to date, the fields stay put',
    'Доказ: під перешкодою пікселі сповзають до діагоналі HV = HH': 'Proof: under interference, pixels slide towards the HV = HH diagonal',
    '108 D (вечірній)': '108 D (evening pass)', '143 A (нічний)': '143 A (night pass)',
    'фіолетовий — відкритий ґрунт': 'violet: bare soil', 'зелений — рослинність або перешкода': 'green: vegetation or interference', 'темний — вода': 'dark: water',
    ', двополяризаційний композит R = HH, G = HV, B = HH − HV, 240 м · під плямою поля «зеленіють»: HV у дБ росте сильніше за HH, і співвідношення каналів руйнується':
        ', dual-pol composite R = HH, G = HV, B = HH − HV, 240 m · under the blob fields turn green: in dB, HV rises more than HH and the channel ratio collapses',
    ', R = HH, G = HV, B = HH − HV · шість дат сезону, однакові діапазони для всіх панелей':
        ', R = HH, G = HV, B = HH − HV · six dates of the season, same ranges for all panels',
    ', щільність пікселів у площині HH–HV · на найменш забрудненій даті HV нижче за HH на ':
        ', pixel density in the HH–HV plane · on the least contaminated date HV is below HH by ',
    ' дБ; однакова перешкода в обох каналах стискає різницю до ': ' dB; equal interference in both channels squeezes the gap to ',
    ' дБ, після очищення — ': ' dB, after cleaning to ',
    ' · найменш забруднена дата, сирий': ' · least contaminated date, raw', ' · сирий знімок': ' · raw image', ' · після очищення': ' · after cleaning',
    ', найменш забруднена дата': ', least contaminated date', ', після очищення': ', after cleaning', ', сирий': ', raw',
    '\nмедіана HH − HV = ': '\nmedian HH − HV = ', 'Рілля, трек ': 'Cropland, track ',
    'NISAR · трек 143 A · ': 'NISAR · track 143 A · ', 'NISAR, трек ': 'NISAR, track ',
    'ОЧИЩЕНИЙ': 'CLEANED', 'СИРИЙ': 'RAW', 'трек ': 'track ',
}
EXACT = {'черв': 'Jun', 'лип': 'Jul', 'серп': 'Aug', 'вер': 'Sep', 'жовт': 'Oct', 'лист': 'Nov'}
_ORDER = sorted(PHRASES, key=len, reverse=True)
_CYR = re.compile('[А-Яа-яІіЇїЄєҐґ]')
LEFTOVER = set()


def tr(s):
    if not isinstance(s, str) or not _CYR.search(s):
        return s
    if s in EXACT:
        return EXACT[s]
    for k in _ORDER:
        if k in s:
            s = s.replace(k, PHRASES[k])
    s = s.replace(' дБ', ' dB').replace('дБ', 'dB').replace(' км²', ' km²').replace(' км', ' km')
    s = re.sub(r'(\d) м\b', r'\1 m', s)
    if _CYR.search(s):
        LEFTOVER.add(s)
    return s


def install():
    import matplotlib.text as mtext
    from PIL import ImageDraw
    orig = mtext.Text.set_text
    mtext.Text.set_text = lambda self, s: orig(self, tr(s) if isinstance(s, str) else s)
    orig_draw = ImageDraw.ImageDraw.text
    ImageDraw.ImageDraw.text = lambda self, xy, text, *a, **k: orig_draw(self, xy, tr(text), *a, **k)
