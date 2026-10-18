"""Two-pass cashier-violation analysis: model extracts facts, code applies the rules.

Pass 1 (overview): the whole cropped clip; the model lists events with timestamps and the
         state of the desk/safe/screen/door when the cashier leaves. No violation judgement.
Pass 2 (detail):   short windows with fine hand motion (windows.py), cut out, zoomed and
         upscaled; the model answers narrow yes/no questions about that window.
Checks:  every candidate from pass 1/2 (safe opened, bag handed over, second client, coffee,
         personal items, door, cashier leaving) triggers a short focused clip at that moment
         with one narrow question; only confirmed candidates become facts.
Rules:   deterministic code turns the facts into the required JSON (verdict, violations).

    python pipeline.py video.mov [--debug]
"""
import argparse
import json
import os
import sys
import time

import av
import numpy as np
from PIL import Image

import infer_video
from crop import find_action_box
from windows import find_windows

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP_DIR = os.path.join(HERE, "cropped", "windows")

# Optional wall-clock deadline (epoch seconds) for the current video. Optional stages are
# skipped as it approaches, so a result is always returned in time.
_DEADLINE = None


def set_deadline(t):
    global _DEADLINE
    _DEADLINE = t


_PROGRESS = None


def set_progress(fn):
    """Optional callback(stage_name) for reporting progress to a UI."""
    global _PROGRESS
    _PROGRESS = fn


def _progress(name):
    if _PROGRESS:
        try:
            _PROGRESS(name)
        except Exception:  # noqa: BLE001 - progress reporting must never break analysis
            pass


def time_left():
    if _DEADLINE is None:
        return 1e9
    end = _DEADLINE() if callable(_DEADLINE) else _DEADLINE
    return end - time.time()

PASS1_PROMPT = """Ты смотришь видео из кассового помещения банка (камера сверху). В помещении работает кассир;
иногда приходят клиенты или инкассатор с зелёной пломбированной сумкой.

Роли: КАССИР — сотрудник банка, который сидит за столом со стороны ноутбука и счётной машинки и
работает с ними. КЛИЕНТ (или инкассатор) приходит к кассе и сидит/стоит НАПРОТИВ кассира.

НЕ оценивай нарушения. Только зафиксируй, что видно, с секундами видео.
Верни ТОЛЬКО JSON по схеме (null — если события не было или не видно):

{
  "observations": "<не более 6 коротких предложений с секундами: кто пришёл/ушёл, что кассир делает с деньгами/документами/сумкой, сейф, ноутбук, дверь. Не повторяйся>",
  "clients": [{"arrive_sec": <число|null>, "leave_sec": <число|null>}],
  "cashier_checked_client_document_before_counting": <true|false|null>,
  "cashier_enters_sec": <число|null>,
  "cashier_starts_work_sec": <число|null>,
  "door_left_open_while_cashier_works_inside": <true|false>,
  "cashier_personal_items_on_desk": <true|false>,
  "sealed_bag": {
    "received_sec": <число|null>,
    "accompanying_paper_received_with_bag": <true|false|null>,
    "opened_sec": <число|null>,
    "cash_put_into_safe_sec": <число|null>
  },
  "safe_opened_sec": <число|null>,
  "safe_left_open_after_use_sec": <число|null>,
  "previous_client_cash_still_on_desk_when_next_client_served_sec": <число|null>,
  "coffee_drinking_with_cash_on_desk_sec": <число|null>,
  "cashier_leaves_sec": <число|null>,
  "at_leave": {
    "cash_on_desk": <true|false|null>,
    "stamps_or_documents_on_desk": <true|false|null>,
    "sealed_bag_on_desk": <true|false|null>,
    "safe_open": <true|false|null>,
    "key_in_safe_lock": <true|false|null>,
    "screen_on_and_visible": <true|false|null>
  }
}

Пояснения:
- "door_left_open_while_cashier_works_inside": true только если кассир вошла, дверь осталась открытой и кассир работает внутри.
  Дверь, которая открыта после ухода кассира, сюда не относится. Закрытая, но не запертая дверь — это закрыта.
- "cashier_personal_items_on_desk": личные вещи сотрудника — сумка, рюкзак, куртка, пальто. Чашка, телефон, бумаги — не считаются.
- "cashier_starts_work_sec": секунда, когда кассир (после входа) села за стол и начала работать.
- "safe_opened_sec": секунда, когда кассир открыла сейф/металлический шкаф (первый раз).
- "safe_left_open_after_use_sec": секунда, когда кассир закончила работу с сейфом/шкафом и отошла или села, а дверца осталась открытой.
- "cashier_checked_client_document_before_counting": клиент принёс деньги — взяла ли кассир его документ/приглашение
  и посмотрела в него ДО начала пересчёта (null, если клиента с деньгами не было).
- "previous_client_cash_still_on_desk_when_next_client_served_sec": секунда, когда кассир начала обслуживать
  следующего клиента, а деньги предыдущего клиента всё ещё лежат на столе (не убраны).
- "cashier_leaves_sec": секунда, когда КАССИР встала со своего места и ушла (из помещения или от стола).
  Уход клиента или инкассатора — НЕ уход кассира: если уходит только клиент, а кассир остаётся сидеть, ставь null.
- "at_leave": состояние сразу после того, как кассир покинула рабочее место (null для всех полей, если не уходила).
  "screen_on_and_visible": экран светится и показывает содержимое (тёмный экран или закрытый ноутбук — false).
"""

PASS2_PROMPT = """Это короткий увеличенный фрагмент видео (секунды {start:.0f}–{end:.0f}): руки кассира над столом.
Ответь только о том, что видно в ЭТОМ фрагменте. Верни ТОЛЬКО JSON:

{{
  "what_happens": "<одно-два предложения: что делают руки кассира и клиента с деньгами и бумагами>",
  "counts_banknotes": <true|false>,
  "gives_banknotes_to_client": <true|false>,
  "looks_at_document": <true|false>,
  "takes_document_from_another_person": <true|false>,
  "checks_banknotes_on_device": <true|false>,
  "combines_cash_piles": <true|false>,
  "client_at_desk": <true|false>,
  "opens_safe": <true|false>,
  "drinks_from_cup": <true|false>,
  "event_sec": <секунда внутри фрагмента, когда происходит главное действие, от 0>
}}

Значения полей:
- "counts_banknotes": кассир пересчитывает купюры — перебирает их по одной или пропускает через счётную машинку.
- "gives_banknotes_to_client": кассир передаёт купюры клиенту в руки или кладёт их перед клиентом.
- "looks_at_document": кассир берёт лист бумаги/документ и смотрит в него.
- "takes_document_from_another_person": кассир принимает лист/документ от другого человека.
- "checks_banknotes_on_device": купюры проверяются на детекторе, под лампой или на просвет.
- "combines_cash_piles": две отдельные пачки денег складываются в одну.
- "client_at_desk": напротив кассира за столом находится другой человек (клиент).
- "opens_safe": кассир открывает дверцу сейфа/металлического шкафа.
- "drinks_from_cup": кассир держит чашку/стакан или пьёт из него.

Поля должны совпадать с "what_happens": если ты описал действие — поставь true в соответствующем поле.
"""


LEAVE_PROMPT = """Это несколько кадров рабочего места кассира сразу после того, как кассир ушла.
Внимательно рассмотри стол, сейф/металлический шкаф и экран компьютера. Верни ТОЛЬКО JSON:

{
  "desk_items": "<перечисли всё, что лежит на столе>",
  "cashier_at_workplace": <true|false>,
  "cash_on_desk": <true|false>,
  "stamps_or_documents_on_desk": <true|false>,
  "sealed_bag_on_desk": <true|false>,
  "safe_open": <true|false>,
  "key_in_safe_lock": <true|false>,
  "laptop_screen": "<bright | dark | closed | not_visible>"
}

- "cashier_at_workplace": КАССИР (сотрудник со стороны ноутбука/счётной машинки) сидит или стоит на своём
  рабочем месте. Клиент напротив кассира не считается.
- "cash_on_desk": на столе лежат купюры/пачка денег (даже небольшая). Деньги могут выглядеть как небольшие
  прямоугольные листки бумаги розового, зелёного, голубого или бежевого цвета.
- "stamps_or_documents_on_desk": на столе лежат штампы (печати) или рабочие документы/бланки.
  Штамп выглядит как небольшой цилиндр или брусок с ручкой (часто чёрный или синий); документ/бланк —
  лист или стопка листов бумаги, папка.
- "sealed_bag_on_desk": на столе лежит зелёная инкассаторская сумка.
- "safe_open": дверца сейфа (серый металлический шкаф) буквально приоткрыта или распахнута — видна щель или внутренняя часть.
- "key_in_safe_lock": из замка на дверце сейфа торчит ключ (маленький металлический предмет у ручки/замка).
- "laptop_screen": цвет экрана ноутбука/монитора: "bright" — экран белый/светлый/цветной, светится;
  "dark" — экран чёрный или тёмно-серый; "closed" — ноутбук закрыт; "not_visible" — экрана не видно.
"""


SAFE_PROMPT = """Увеличенный фрагмент: сейф — серый металлический шкаф рядом со столом кассира.
Оцени ПОСЛЕДНИЕ кадры фрагмента. Верни ТОЛЬКО JSON:
{"what_happens": "<одно предложение>", "safe_door_open": <true|false>, "cashier_at_safe_working_with_it": <true|false>}
- "safe_door_open": дверца сейфа буквально приоткрыта или распахнута — она не прилегает к корпусу:
  видна щель между дверцей и корпусом, дверца повёрнута наружу или видна внутренняя часть шкафа.
  Если передняя стенка шкафа ровная и закрытая — false.
- "cashier_at_safe_working_with_it": кассир стоит у сейфа и кладёт/берёт что-то из него.
"""

BAG_PROMPT = """Короткий фрагмент: человек передаёт кассиру зелёную инкассаторскую сумку. Верни ТОЛЬКО JSON:
{"what_happens": "<одно предложение>", "green_bag_handed_to_cashier": <true|false>, "paper_or_document_handed_over": <true|false>}
- "green_bag_handed_to_cashier": в этом фрагменте человек действительно передаёт кассиру зелёную сумку.
- "paper_or_document_handed_over": вместе с сумкой (или сразу до/после) кассиру передают лист бумаги/документ,
  либо кассир берёт и смотрит такой лист.
"""

# Neutral (non-leading) question: the model has to name what is handed over, instead of confirming "a bag".
BAG_PRESENCE_PROMPT = """Короткий фрагмент видео кассы. Посмотри, что посетитель передаёт кассиру или кладёт на стол,
и что лежит на столе кассира. Верни ТОЛЬКО JSON:
{"what_happens": "<одно предложение>", "handed_item": "<banknotes|bag|documents|nothing|other>", "cloth_bag_on_desk": <true|false>}
- "handed_item": что именно передают кассиру. "banknotes" — купюры или пачка денег (в руке, в пачке, в конверте);
  "bag" — сумка или мешок из ткани (например, инкассаторская); "documents" — листы бумаги; "nothing" — ничего не передают.
- "cloth_bag_on_desk": на столе кассира лежит сумка или мешок из ткани (не ноутбук, не машинка, не пачка денег).
"""


def bag_present(ans):
    return str(ans.get("handed_item", "")).strip().lower() == "bag" or ans.get("cloth_bag_on_desk") is True


BAG_OPEN_PROMPT = """Короткий фрагмент рабочего места кассира. Верни ТОЛЬКО JSON:
{"what_happens": "<одно предложение>", "green_bag_opened_and_cash_taken_out": <true|false>}
- "green_bag_opened_and_cash_taken_out": кассир открывает зелёную инкассаторскую сумку и достаёт из неё деньги.
"""

COFFEE_PROMPT = """Фрагмент рабочего места кассира. Верни ТОЛЬКО JSON:
{"what_happens": "<одно предложение>", "cashier_drinks_from_cup": <true|false>, "banknotes_lying_on_desk": <true|false>, "event_sec": <секунда внутри фрагмента, от 0>}
- "cashier_drinks_from_cup": кассир пьёт кофе/чай — держит чашку или стакан в руках или подносит ко рту.
- "banknotes_lying_on_desk": в это время на столе открыто лежат купюры (в любом месте стола, не в сейфе и не в ящике).
  Купюры выглядят как небольшие прямоугольные листки розового, зелёного, голубого или бежевого цвета.
"""

PERSONAL_PROMPT = """Несколько кадров рабочего стола кассира. Верни ТОЛЬКО JSON:
{"desk_items": "<перечисли, что лежит на столе>", "handbag_or_backpack_on_desk": <true|false>, "jacket_or_coat_on_desk": <true|false>}
- Чашка, телефон, бумаги, техника — НЕ сумка и НЕ одежда.
"""

DOOR_PROMPT = """Увеличенные кадры входной двери кассового помещения (стеклянная дверь с металлической рамой и ручкой),
кассир уже работает за столом. Верни ТОЛЬКО JSON:
{"what_happens": "<одно предложение>", "door_open": <true|false>}
- "door_open": дверь буквально открыта или приоткрыта — створка не прилегает к раме: видна щель/проём
  между створкой и рамой, створка повёрнута внутрь или наружу. Закрытая (даже не запертая) дверь — false.
"""

CASH_PROMPT = """Увеличенные кадры поверхности стола кассира после того, как кассир ушла. Верни ТОЛЬКО JSON:
{"what_on_desk": "<перечисли предметы на столе>", "banknotes_on_desk": <true|false>}
- "banknotes_on_desk": на столе лежат купюры — одна или несколько: небольшие прямоугольные листки
  розового, зелёного, голубого, бежевого или белого цвета, иногда стопкой. Чеки и документы обычно крупнее и белые.
"""

KEY_PROMPT = """Увеличенные кадры дверцы сейфа (серый металлический шкаф) после того, как кассир ушла. Верни ТОЛЬКО JSON:
{"what_near_lock": "<одно предложение: что видно у замка/ручки дверцы>", "key_in_lock": <true|false>}
- "key_in_lock": из замка на дверце сейфа торчит ключ — небольшой металлический или тёмный предмет,
  выступающий из замочной скважины рядом с ручкой (иногда с брелоком/кольцом). Пустая замочная скважина — false.
"""

NO_CLIENT_PROMPT = """Фрагмент сразу после того, как клиент мог уйти от кассы. Верни ТОЛЬКО JSON:
{"what_happens": "<одно предложение>", "client_at_desk": <true|false>, "cashier_counts_money": <true|false>}
- "client_at_desk": напротив кассира за столом сидит/стоит клиент.
- "cashier_counts_money": кассир пересчитывает купюры руками или на счётной машинке.
"""

SCREEN_PROMPT = """Увеличенные кадры ноутбука на столе кассира после того, как кассир ушла. Верни ТОЛЬКО JSON:
{"what_on_screen": "<одно предложение>", "screen": "<content | lock_screen | dark | closed | not_visible>"}
- "content": на экране открыта рабочая программа — белое/светлое окно, таблица, текст, документ, рабочий стол с окнами.
  Компьютер НЕ заблокирован.
- "lock_screen": экран блокировки Windows — во весь экран фотография/картинка (часто синий пейзаж),
  иногда с часами или именем пользователя. Компьютер заблокирован.
- "dark": экран чёрный/тёмный (выключен или спит).
- "closed": крышка ноутбука закрыта.
- "not_visible": экрана не видно.
"""

DOOR_COMPARE_PROMPT = """На каждом кадре две половины. СЛЕВА — входная дверь кассового помещения ДО того, как кассир вошла
(дверь закрыта). СПРАВА — та же дверь с той же точки, когда кассир уже вошла и работает.
Сравни положение створки двери справа с положением слева. Верни ТОЛЬКО JSON:
{"difference": "<одно предложение: что изменилось в области двери>", "door_open": <true|false>}
- "door_open": true, если справа створка двери повёрнута/распахнута или приоткрыта по сравнению с закрытой дверью слева
  (например, на месте прозрачного стекла с видом в коридор появилась тёмная повернутая створка, видна щель или проём).
  Если дверь справа в том же положении, что слева, — false. Человек или мебель перед дверью — не изменение двери.
"""

DOC_CHECK_PROMPT = """Фрагмент: клиент пришёл к кассе с деньгами, кассир начинает работу с ним. Верни ТОЛЬКО JSON:
{"what_happens": "<одно-два предложения>", "cashier_takes_and_reads_paper_from_client": <true|false>}
- "cashier_takes_and_reads_paper_from_client": кассир берёт у клиента лист/бланк/приглашение/папку с бумагами
  и смотрит в него (до или в начале пересчёта денег).
"""

# Neutral direction question: "the cashier gives money to the client" is often reported for a client depositing it.
TRANSFER_PROMPT = """Короткий фрагмент у кассы. КАССИР — сотрудник со стороны ноутбука и счётной машинки;
КЛИЕНТ сидит или стоит напротив кассира. Посмотри, кто кому передаёт деньги. Верни ТОЛЬКО JSON:
{"what_happens": "<одно предложение>", "money_moves": "<client_to_cashier|cashier_to_client|none|unclear>"}
- "client_to_cashier": клиент отдаёт деньги кассиру или кладёт их на стол, а кассир их берёт.
- "cashier_to_client": кассир отдаёт купюры клиенту в руки или пододвигает их к клиенту, и клиент их забирает.
- "none": деньги никто никому не передаёт.
"""


def returns_money(ans):
    return str(ans.get("money_moves", "")).strip().lower() == "cashier_to_client"


AFTER_COUNT_PROMPT = """Фрагмент сразу после того, как кассир пересчитала деньги. Посмотри, что кассир делает
с пересчитанными купюрами. Верни ТОЛЬКО JSON:
{"what_happens": "<одно предложение>", "next_action": "<checks_authenticity|stacks_or_bundles|gives_to_client|puts_away|still_counting|other>", "event_sec": <секунда от начала фрагмента>}
- "checks_authenticity": проверяет купюры на подлинность — подносит к детектору или УФ-лампе, смотрит на просвет.
  Пропуск купюр через счётную машинку — это пересчёт, а не проверка подлинности.
- "stacks_or_bundles": складывает пересчитанные купюры в одну стопку или пачку, выравнивает, перетягивает.
- "gives_to_client": отдаёт деньги клиенту. "puts_away": убирает в сейф, ящик или кассу.
- "still_counting": продолжает пересчитывать.
"""


def after_count_action(ans):
    return str(ans.get("next_action", "")).strip().lower()


def _before_counting(w, p2):
    """A handover seen before any counting is the client's deposit, not a return of counted money."""
    def counting(x):
        return x["ans"].get("counts_banknotes") is True or x["ans"].get("checks_banknotes_on_device") is True
    starts = [x["start"] for x in p2 if counting(x)]
    return bool(starts) and not counting(w) and w["t"] < min(starts)


NEXT_CLIENT_PROMPT = """Фрагмент: к кассе подходит следующий клиент, пока у кассира может быть операция предыдущего клиента.
Верни ТОЛЬКО JSON:
{
  "what_happens": "<два-три предложения>",
  "takes_paper_from_new_client": <true|false>,
  "previous_counting_still_ongoing": <true|false>,
  "previous_client_money_left_on_desk": <true|false>,
  "money_of_two_clients_combined": <true|false>,
  "event_sec": <секунда внутри фрагмента, от 0>
}
- "takes_paper_from_new_client": кассир берёт лист/документ у нового клиента.
- "previous_counting_still_ongoing": В ТОТ МОМЕНТ, когда кассир берёт лист нового клиента, она держит в руках
  или пересчитывает деньги предыдущего клиента (пересчёт не закончен).
- "previous_client_money_left_on_desk": пересчёт предыдущего клиента закончен, но его деньги просто лежат на столе
  (не убраны в сейф/ящик), а кассир уже занимается новым клиентом.
- "money_of_two_clients_combined": деньги нового клиента кладутся в ту же стопку, что и деньги предыдущего.
"""


def cut_clip(src, box, start, end, out, target_pixels=250_000, max_fps=8.0):
    """Write the [start, end] part of src, cropped to box and upscaled to ~target_pixels.

    Only every step-th frame is written (output rate src_fps/step <= max_fps): timestamps stay the
    same and re-encoding long videos stays cheap; the model samples 1-6 fps anyway."""
    x0, y0, x1, y1 = box
    scale = max(1.0, (target_pixels / ((x1 - x0) * (y1 - y0))) ** 0.5)
    w, h = int((x1 - x0) * scale) // 2 * 2, int((y1 - y0) * scale) // 2 * 2
    from fractions import Fraction
    with av.open(src) as inp, av.open(out, "w") as dst:
        s = inp.streams.video[0]
        fps = float(s.average_rate or 30)
        step = max(1, int(fps // max_fps))
        o = dst.add_stream("libx264", rate=Fraction(s.average_rate or 30) / step)
        o.width, o.height, o.pix_fmt = w, h, "yuv420p"
        o.options = {"crf": "16", "preset": "veryfast"}
        for i, frame in enumerate(inp.decode(s)):
            t = i / fps
            if t < start:
                continue
            if t > end:
                break
            if (i - int(start * fps)) % step:
                continue
            img = frame.to_image().crop(box).resize((w, h), Image.LANCZOS)
            for p in o.encode(av.VideoFrame.from_ndarray(np.asarray(img), format="rgb24")):
                dst.mux(p)
        for p in o.encode():
            dst.mux(p)
    return out


def _ask_json(model, processor, path, prompt, fps, max_frames, max_pixels, max_new_tokens):
    answer, _ = infer_video.ask_video(model, processor, path, prompt, None, fps, max_frames,
                                      max_pixels, max_new_tokens, repetition_penalty=1.1)
    obj = _parse(answer)
    return obj, answer


def _parse(text):
    import re
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- rules

def apply_rules(p1, p2, checks=None):
    """Facts -> list of violations in the required format, ordered by time.

    checks: results of the focused clips (see run_checks); a pass-1 claim that has a focused
    check only counts when the check confirms it."""
    p1, checks = p1 or {}, checks or {}
    v = []

    def add(t, typ, desc, ev=1):
        # ev: 2 = confirmed by two independent focused looks, 1 = one look / window, 0 = overview only
        v.append({"type": typ, "timecode_sec": round(t or 0.0, 1), "description": desc, "_ev": ev})

    def ev_of(check):
        if check and check.get("_weak"):
            return 0
        return 2 if check and check.get("second") and not check["ans"].get("_unconfirmed") else 1
    ev_leave = 2 if p1.get("_leave_double") else 1
    leave_weak = set(p1.get("_leave_weak") or [])

    clients = [c for c in (p1.get("clients") or []) if isinstance(c, dict)]
    if not clients and any(w["ans"].get("client_at_desk") is True for w in p2):
        clients = [{"arrive_sec": None, "leave_sec": None}]
    bag = p1.get("sealed_bag") or {}
    at = p1.get("at_leave") or {}
    leave = _num(p1.get("cashier_leaves_sec"))
    enters = _num(p1.get("cashier_enters_sec"))
    starts = _num(p1.get("cashier_starts_work_sec"))
    bag_rx, bag_open, bag_safe = _num(bag.get("received_sec")), _num(bag.get("opened_sec")), _num(bag.get("cash_put_into_safe_sec"))
    has_bag = any(x is not None for x in (bag_rx, bag_open, bag_safe)) or at.get("sealed_bag_on_desk") is True
    bp = checks.get("bag_presence")
    bag_refuted = bool(bp) and not (bag_present(bp["ans"]) and not bp.get("_weak"))
    if bag_refuted:   # the neutral look saw no bag: pass-1 bag claims (and bag-only logic) are dropped
        has_bag = False
        at = {**at, "sealed_bag_on_desk": False}

    # --- leaving the workplace (at_leave is overwritten by the leave snapshot when it ran)
    if leave is not None:
        cash_c = checks.get("cash")
        if cash_c and cash_c["ans"].get("banknotes_on_desk") is True and at.get("cash_on_desk") is not True:
            add(leave, "CASH_NOT_SECURED_BEFORE_LEAVING", "Кассир покинула рабочее место, оставив наличные на столе.",
                ev_of(cash_c))
        elif at.get("cash_on_desk") is True:
            add(leave, "CASH_NOT_SECURED_BEFORE_LEAVING", "Кассир покинула рабочее место, оставив наличные на столе.", 0 if "cash_on_desk" in leave_weak else ev_leave)
        if at.get("stamps_or_documents_on_desk") is True:
            add(leave, "DOCUMENTS_OR_STAMPS_NOT_SECURED", "Кассир покинула рабочее место, оставив штампы/документы вне сейфа.", 0 if "stamps_or_documents_on_desk" in leave_weak else ev_leave)
        if at.get("sealed_bag_on_desk") is True:
            add(leave, "SEALED_BAG_LEFT_UNATTENDED", "Кассир ушла, оставив инкассаторскую сумку на столе.", 0 if "sealed_bag_on_desk" in leave_weak else ev_leave)
        kc = checks.get("key")
        if kc is not None:
            if kc["ans"].get("key_in_lock") is True:
                # The key is a few pixels: one look is weak evidence, only two agreeing looks confirm it.
                add(leave, "SAFE_KEY_LEFT_IN_LOCK", "Кассир покинула рабочее место, оставив ключ в замке сейфа.",
                    2 if ev_of(kc) == 2 else 0)
        elif at.get("key_in_safe_lock") is True:
            add(leave, "SAFE_KEY_LEFT_IN_LOCK", "Кассир покинула рабочее место, оставив ключ в замке сейфа.", 0 if "key_in_safe_lock" in leave_weak else ev_leave)
        sc_ = checks.get("screen")   # unlocked = a work window on screen; lock screen / dark / closed are fine
        if sc_ and str(sc_["ans"].get("screen", "")).strip().lower() == "content":
            add(leave, "COMPUTER_NOT_SECURED", "Кассир покинула рабочее место, не заблокировав компьютер (экран остаётся доступным).", ev_of(sc_))

    # --- safe left open after use (confirmed a few seconds after it was opened, or at leave)
    sc = checks.get("safe")
    if sc and sc["ans"].get("safe_door_open") is True and sc["ans"].get("cashier_at_safe_working_with_it") is not True:
        add(sc["t"], "SAFE_LEFT_OPEN", "Кассир закончила работу с сейфом и отошла, оставив его открытым.", ev_of(sc))
    elif leave is not None and at.get("safe_open") is True:
        add(leave, "SAFE_LEFT_OPEN", "Кассир покинула рабочее место, оставив сейф открытым.", 0 if "safe_open" in leave_weak else ev_leave)

    # --- start of work: door, personal items, coffee (pass-1 claim + focused confirmation)
    t_work = starts if starts is not None else (enters if enters is not None else 0.0)
    dc = checks.get("door")
    if enters is not None and dc and dc["ans"].get("door_open") is True:
        add(t_work, "DOOR_LEFT_OPEN", "Кассир вошла и приступила к работе, не закрыв за собой дверь.", ev_of(dc))
    pc = checks.get("personal")
    if pc and (pc["ans"].get("handbag_or_backpack_on_desk") is True or pc["ans"].get("jacket_or_coat_on_desk") is True):
        add(t_work, "PERSONAL_ITEMS_ON_WORK_DESK", "На рабочем столе во время работы лежат личные вещи кассира (сумка/одежда).", ev_of(pc))
    cc = checks.get("coffee")
    if cc and cc["ans"].get("cashier_drinks_from_cup") is True and cc["ans"].get("banknotes_lying_on_desk") is True:
        add(cc["t"], "CASH_EXPOSED_DURING_COFFEE", "Кассир пьёт кофе рядом с открыто лежащими на столе наличными.", ev_of(cc))

    # --- sealed bag
    bc = checks.get("bag")
    if (bc and bc["ans"].get("green_bag_handed_to_cashier") is True
            and bc["ans"].get("paper_or_document_handed_over") is False):
        add(bc["t"], "SEALED_BAG_ACCEPTED_WITHOUT_DOCUMENT", "Кассир приняла пломбированную сумку без препроводительного документа.", ev_of(bc))
    bo = checks.get("bag_open")
    if bag_safe is not None and bo and bo["ans"].get("green_bag_opened_and_cash_taken_out") is True:
        lo = bag_open if bag_open is not None else (bag_rx if bag_rx is not None else 0.0)
        counted = any(w["ans"].get("counts_banknotes") is True and lo - 1 <= w["t"] <= bag_safe + 1 for w in p2)
        mc = checks.get("machine")
        if mc and mc["ans"].get("machine_running") is True:   # banknote counter was busy in that interval
            counted = True
        windows_cover = any(lo - 1 <= w["start"] and w["end"] <= bag_safe + 1 for w in p2)
        if not counted and windows_cover:
            add(bag_safe, "SEALED_BAG_CASH_NOT_COUNTED", "Кассир извлекла деньги из пломбированной сумки и поместила их в сейф без пересчёта.", ev_of(bo))

    # --- client operations
    if clients and not has_bag:
        counting = sorted(w["t"] for w in p2 if w["ans"].get("counts_banknotes") is True)
        first_count = counting[0] if counting else None
        gives = [w for w in sorted(p2, key=lambda w: w["t"]) if w["ans"].get("gives_banknotes_to_client") is True
                 and not _before_counting(w, p2)]
        tr = checks.get("transfer")
        if tr is not None and not returns_money(tr["ans"]):
            gives = []          # the neutral look saw the client handing money over (or nothing), not a return
        strong = [w for w in gives if w["ans"].get("counts_banknotes") is True and not (tr or {}).get("_weak")]
        if strong:
            add(strong[0]["t"], "COUNTED_CASH_RETURNED", "Кассир в процессе пересчёта вернула клиенту часть принятых денег.")
        elif gives:   # handed back, counting not confirmed: weak unless the direction check confirmed it twice
            add(gives[0]["t"], "COUNTED_CASH_RETURNED", "Кассир вернула клиенту часть принятых денег.",
                min(1, ev_of(tr)) if tr is not None else 0)
        dk = checks.get("doc_check")
        seen_doc = any(w["ans"].get("looks_at_document") is True and w["t"] <= first_count + 1 for w in p2) \
            if first_count is not None else False
        if first_count is not None and not seen_doc:
            if dk and dk["ans"].get("cashier_takes_and_reads_paper_from_client") is False:
                add(first_count, "CASH_COUNTING_WITHOUT_DOCUMENT_CHECK",
                    "Кассир начала пересчитывать деньги, не проверив документ/приглашение.", ev_of(dk))
            elif not (dk and dk["ans"].get("cashier_takes_and_reads_paper_from_client") is True and not dk.get("_weak")):
                add(first_count, "CASH_COUNTING_WITHOUT_DOCUMENT_CHECK",
                    "Кассир начала пересчитывать деньги, не проверив документ/приглашение.", 0)
        combined = next((w for w in sorted(p2, key=lambda w: w["t"])
                         if w["ans"].get("combines_cash_piles") is True), None)
        nc = checks.get("next_client")
        if len(clients) >= 2 and nc:
            a = nc["ans"]
            # Did the first client's operation overlap the second client's arrival? If the first client
            # had already left, "starting the next client too early" is impossible; leftover handling of
            # the first client's money then means their cash was not secured.
            by_arrival = sorted((c for c in clients if _num(c.get("arrive_sec")) is not None),
                                key=lambda c: _num(c.get("arrive_sec")))
            overlap = True
            if len(by_arrival) >= 2:
                first_leave = _num(by_arrival[0].get("leave_sec"))
                second_arrive = _num(by_arrival[1].get("arrive_sec"))
                if first_leave is not None and first_leave < second_arrive + 1:
                    overlap = False
            if not overlap and a.get("previous_counting_still_ongoing") is True:
                a = {**a, "previous_counting_still_ongoing": False, "previous_client_money_left_on_desk": True}
            if a.get("money_of_two_clients_combined") is True:
                add(nc["t"], "CLIENTS_CASH_MIXED", "Кассир объединила деньги двух разных клиентов в одну стопку.", ev_of(nc))
            if a.get("takes_paper_from_new_client") is True and a.get("previous_counting_still_ongoing") is True:
                add(nc["t"], "NEXT_CLIENT_STARTED_BEFORE_PREVIOUS_COMPLETED",
                    "Кассир приняла документ следующего клиента, не закончив операцию предыдущего.", ev_of(nc))
            elif a.get("previous_client_money_left_on_desk") is True:
                add(nc["t"], "PREVIOUS_CLIENT_CASH_NOT_SECURED",
                    "Кассир начала обслуживать следующего клиента, хотя наличные предыдущего клиента остались на столе.", ev_of(nc))
        # counting continues after the client has left
        ncc = checks.get("no_client")
        first_with_client = min((w["t"] for w in p2 if w["ans"].get("client_at_desk") is True), default=None)
        alone_counting = [w["t"] for w in p2 if w["ans"].get("counts_banknotes") is True
                          and w["ans"].get("client_at_desk") is False
                          and first_with_client is not None and w["t"] > first_with_client]
        if (ncc and ncc["ans"].get("client_at_desk") is False and ncc["ans"].get("cashier_counts_money") is True):
            add(ncc["t"], "CASH_COUNTED_WITHOUT_CLIENT_PRESENT", "Клиент ушёл, а кассир продолжает пересчёт его денег.", ev_of(ncc))
        elif alone_counting:
            add(alone_counting[0], "CASH_COUNTED_WITHOUT_CLIENT_PRESENT", "Клиент ушёл, а кассир продолжает пересчёт его денег.")
        else:
            last_leave = max((_num(c.get("leave_sec")) or -1) for c in clients)
            if last_leave >= 0 and any(t > last_leave + 1 for t in counting):
                add(min(t for t in counting if t > last_leave + 1), "CASH_COUNTED_WITHOUT_CLIENT_PRESENT",
                    "Клиент ушёл, а кассир продолжает пересчёт его денег.", 0)
        # Authenticity: only when counted cash is visibly combined/put together and no check was seen.
        if (len(clients) < 2 and combined and counting
                and not any(w["ans"].get("checks_banknotes_on_device") is True for w in p2)):
            add(combined["t"], "CASH_AUTHENTICITY_NOT_CHECKED",
                "После пересчёта кассир сложила купюры в одну стопку, не проверив их на подлинность.")

    # Authenticity: counted banknotes go straight into a stack without a detector/lamp check (client or not).
    ac = checks.get("after_count")
    if ac and after_count_action(ac["ans"]) == "stacks_or_bundles":
        add(ac["t"], "CASH_AUTHENTICITY_NOT_CHECKED",
            "После пересчёта кассир сложила купюры в стопку, не проверив их на подлинность.", ev_of(ac))

    for typ, c in (checks.get("candidates") or {}).items():
        if c["ans"].get("visible") is True:
            add(c["t"], typ, c.get("description") or "", min(1, ev_of(c)))

    # Logical consistency: a type is only possible if its scenario happened in this clip.
    counting_seen = (any(w["ans"].get("counts_banknotes") is True for w in p2)
                     or ((checks.get("machine") or {}).get("ans") or {}).get("machine_running") is True)
    bag_seen = not bag_refuted and (
        has_bag or ((checks.get("bag") or {}).get("ans") or {}).get("green_bag_handed_to_cashier") is True
        or ((checks.get("bag_open") or {}).get("ans") or {}).get("green_bag_opened_and_cash_taken_out") is True)
    safe_used = (checks.get("safe") is not None or at.get("safe_open") is True
                 or _num(p1.get("safe_opened_sec")) is not None)
    n_clients = len(clients)
    # A return needs a handover that is not the deposit and that the direction check did not refute.
    any_give = [w for w in p2 if w["ans"].get("gives_banknotes_to_client") is True]
    tr = checks.get("transfer")
    return_ok = not (any_give and all(_before_counting(w, p2) for w in any_give)) \
        and not (tr is not None and not returns_money(tr["ans"]))
    possible = {
        "COUNTED_CASH_RETURNED": n_clients >= 1 and return_ok,
        "CASH_COUNTING_WITHOUT_DOCUMENT_CHECK": n_clients >= 1 and counting_seen,
        "CASH_COUNTED_WITHOUT_CLIENT_PRESENT": n_clients >= 1 and counting_seen,
        "CASH_AUTHENTICITY_NOT_CHECKED": counting_seen and not (
            ac is not None and after_count_action(ac["ans"]) == "checks_authenticity"),
        "PREVIOUS_CLIENT_CASH_NOT_SECURED": n_clients >= 2,
        "CLIENTS_CASH_MIXED": n_clients >= 2,
        "NEXT_CLIENT_STARTED_BEFORE_PREVIOUS_COMPLETED": n_clients >= 2,
        "SEALED_BAG_ACCEPTED_WITHOUT_DOCUMENT": bag_seen,
        "SEALED_BAG_CASH_NOT_COUNTED": bag_seen,
        "SEALED_BAG_LEFT_UNATTENDED": bag_seen and leave is not None,
        "CASH_NOT_SECURED_BEFORE_LEAVING": leave is not None,
        "DOCUMENTS_OR_STAMPS_NOT_SECURED": leave is not None,
        "COMPUTER_NOT_SECURED": leave is not None,
        "SAFE_KEY_LEFT_IN_LOCK": leave is not None or safe_used,
        "SAFE_LEFT_OPEN": safe_used or leave is not None,
        "DOOR_LEFT_OPEN": enters is not None,
    }
    v = [x for x in v if possible.get(x["type"], True)]

    v.sort(key=lambda x: x["timecode_sec"])
    seen, out = set(), []
    for x in v:
        if x["type"] not in seen:
            seen.add(x["type"])
            out.append(x)
    return out


# ---------------------------------------------------------------- driver

END_PROMPT = LEAVE_PROMPT.replace(
    "Это несколько кадров рабочего места кассира сразу после того, как кассир ушла.",
    "Это последние кадры видео с рабочим местом кассира.").replace(
    '  "laptop_screen":', '  "cashier_present_at_desk": <true|false>,\n  "laptop_screen":')

# When several violations are confirmed in one clip, keep the most specific one (scenario-specific
# evidence first, generic state checks last). Only used when cfg["single_violation"] is on.
PRIORITY = [
    "SEALED_BAG_CASH_NOT_COUNTED", "SEALED_BAG_ACCEPTED_WITHOUT_DOCUMENT", "SEALED_BAG_LEFT_UNATTENDED",
    "CLIENTS_CASH_MIXED", "NEXT_CLIENT_STARTED_BEFORE_PREVIOUS_COMPLETED", "PREVIOUS_CLIENT_CASH_NOT_SECURED",
    "CASH_COUNTED_WITHOUT_CLIENT_PRESENT", "COUNTED_CASH_RETURNED", "CASH_AUTHENTICITY_NOT_CHECKED",
    "CASH_COUNTING_WITHOUT_DOCUMENT_CHECK", "CASH_EXPOSED_DURING_COFFEE",
    # leaving-state types, most reliably visible first (a lit screen or bills beat a few-pixel key)
    "CASH_NOT_SECURED_BEFORE_LEAVING", "COMPUTER_NOT_SECURED", "DOCUMENTS_OR_STAMPS_NOT_SECURED",
    "SAFE_LEFT_OPEN", "SAFE_KEY_LEFT_IN_LOCK",
    "DOOR_LEFT_OPEN", "PERSONAL_ITEMS_ON_WORK_DESK", "OTHER_VISIBLE_VIOLATION",
]

DEFAULT_CFG = {"p1_fps": 1.0, "p1_frames": 32, "p1_pixels": 150_528, "p2_fps": 6.0, "p2_frames": 32,
               "p2_pixels": 250_880, "windows": 3, "single_violation": True, "double_check": True,
               "second_opinion": True, "recall_mode": True}


def _check(model, processor, path, box, vid, name, t0, t1, prompt, dur, fps=2.0, frames=12, pixels=401_408):
    """Run one focused question on [t0, t1] of the cropped video; returns dict(t, ans) or None."""
    t0, t1 = max(0.0, t0), min(dur, t1)
    if t1 - t0 < 0.8 or time_left() < 12:
        return None
    clip = cut_clip(path, box, t0, t1, os.path.join(CLIP_DIR, f"{vid}.{name}.mp4"), target_pixels=pixels)
    ans, raw = _ask_json(model, processor, clip, prompt, fps, frames, pixels, 300)
    if not ans:
        return {"t": round(t0, 1), "ans": {}, "raw": raw[:300]}
    rel = _num(ans.get("event_sec"))
    t = t0 + rel if rel is not None and 0 <= rel <= t1 - t0 else t0
    return {"t": round(t, 1), "ans": ans}


GROUND_OBJECT_PROMPT = """Найди на кадре: {what}.
Верни ТОЛЬКО JSON: {{"bbox_2d": [x1, y1, x2, y2]}} — координаты в диапазоне 0–1000 относительно кадра
(0,0 — левый верхний угол). Если объекта не видно, верни {{"bbox_2d": null}}."""


def ground(model, processor, path, box, vid, name, what, t, dur, margin=0.6):
    """Locate an object near time t; return a zoom box in original-pixel coordinates (with margin),
    or None. Qwen3-VL reports boxes on a 0-1000 grid relative to the frame it was shown."""
    t0, t1 = max(0.0, t - 1), min(dur, t + 1)
    if t1 - t0 < 0.5:
        return None
    clip = cut_clip(path, box, t0, t1, os.path.join(CLIP_DIR, f"{vid}.ground_{name}.mp4"), target_pixels=401_408)
    ans, _ = _ask_json(model, processor, clip, GROUND_OBJECT_PROMPT.format(what=what), 1.0, 2, 401_408, 80)
    bb = (ans or {}).get("bbox_2d")
    if not (isinstance(bb, list) and len(bb) == 4 and all(isinstance(x, (int, float)) for x in bb)):
        return None
    bx0, by0, bx1, by1 = box
    bw, bh = bx1 - bx0, by1 - by0
    x0, y0, x1, y1 = [max(0, min(1000, v)) / 1000 for v in bb]
    if x1 - x0 < 0.03 or y1 - y0 < 0.03:
        return None
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, x1 = max(0.0, x0 - mx), min(1.0, x1 + mx)
    y0, y1 = max(0.0, y0 - my), min(1.0, y1 + my)
    return (int(bx0 + x0 * bw), int(by0 + y0 * bh), int(bx0 + x1 * bw), int(by0 + y1 * bh))


def side_by_side_clip(src, box, t_ref, t0, t1, out, target_height=448):
    """Video whose frames show the box at t_ref (left) next to the box at each moment of [t0, t1] (right)."""
    x0, y0, x1, y1 = box
    scale = target_height / (y1 - y0)
    w, h = int((x1 - x0) * scale) // 2 * 2, int(target_height) // 2 * 2
    with av.open(src) as inp, av.open(out, "w") as dst:
        s = inp.streams.video[0]
        fps = float(s.average_rate or 30)
        o = dst.add_stream("libx264", rate=s.average_rate)
        o.width, o.height, o.pix_fmt = 2 * w + 8, h, "yuv420p"
        o.options = {"crf": "16", "preset": "veryfast"}
        ref = None
        for i, frame in enumerate(inp.decode(s)):
            t = i / fps
            if ref is None and t >= t_ref:
                ref = frame.to_image().crop(box).resize((w, h), Image.LANCZOS)
            if t < t0 or ref is None:
                continue
            if t > t1:
                break
            canvas = Image.new("RGB", (2 * w + 8, h), "white")
            canvas.paste(ref, (0, 0))
            canvas.paste(frame.to_image().crop(box).resize((w, h), Image.LANCZOS), (w + 8, 0))
            for p in o.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")):
                dst.mux(p)
        for p in o.encode():
            dst.mux(p)
    return out


def last_motion_time(path, box):
    """Last moment with noticeable motion in the action box (e.g. the cashier walking out)."""
    from windows import _samples
    times, frames = _samples(path, box, 4.0)
    last = None
    for k in range(1, len(frames)):
        d = frames[k] - frames[k - 1]
        m = (np.abs(d - np.median(d)) > 12).mean()
        if 0.01 < m < 0.5:          # some motion, but not a whole-frame lighting change
            last = times[k]
    return last


def run_checks(model, processor, path, box, vid, p1, p2, dur, double_check=True, recall=True):
    p1 = p1 or {}
    checks = {}

    def ask(name, t0, t1, prompt, zoom=None, **kw):
        return _check(model, processor, path, zoom or box, vid, name, t0, t1, prompt, dur, **kw)

    def confirm(name, t0, t1, prompt, positive, negate, shift=1.5, **kw):
        """Ask; if the answer is positive, ask again on a shifted window and require agreement.
        On disagreement the positive fields are overwritten with `negate`."""
        first = ask(name, t0, t1, prompt, **kw)
        if first and double_check and positive(first["ans"]):
            second = ask(name + "_2", t0 + shift, t1 + shift, prompt, **kw)
            if second is None:
                second = ask(name + "_2", t0 - shift, t1 - shift, prompt, **kw)
            first["second"] = second and second["ans"]
            if not (second and positive(second["ans"])):
                if recall:      # misses cost more than false alarms: keep it, ranked lowest
                    first["_weak"] = True
                else:
                    first["ans"] = {**first["ans"], **negate, "_unconfirmed": True}
        return first

    # Leaving the workplace first: the most common violations (cash, screen, key) must not be the
    # ones dropped when the time budget runs out.
    leave = _num(p1.get("cashier_leaves_sec"))
    if leave is not None and leave + 1 < dur:
        x0, y0, x1, y1 = box
        desk_box = (x0, int(y0 + 0.35 * (y1 - y0)), x1, y1)      # lower part of the action area = desk
        checks["cash"] = confirm("cash", leave + 0.5, leave + 4, CASH_PROMPT,
                                 lambda a: a.get("banknotes_on_desk") is True, {"banknotes_on_desk": None},
                                 shift=2.5, frames=6, zoom=desk_box, pixels=401_408)
        if checks.get("cash"):
            checks["cash"]["zoom_box"] = desk_box
        safe_zoom = ground(model, processor, path, box, vid, "safe_leave",
                           "дверца серого металлического шкафа (сейфа) с замком и ручкой",
                           min(dur - 1, leave + 2), dur, margin=0.35)
        if safe_zoom:
            checks["key"] = confirm("key", leave + 0.5, leave + 4, KEY_PROMPT,
                                    lambda a: a.get("key_in_lock") is True, {"key_in_lock": None},
                                    shift=2.5, frames=6, zoom=safe_zoom)
            if checks.get("key"):
                checks["key"]["zoom_box"] = safe_zoom
        laptop_box = ground(model, processor, path, box, vid, "laptop", "ноутбук на столе кассира (экран и клавиатура)",
                            min(dur - 1, leave + 2), dur, margin=0.5)
        checks["screen"] = confirm("screen", leave + 0.5, leave + 4, SCREEN_PROMPT,
                                   lambda a: str(a.get("screen", "")).strip().lower() == "content",
                                   {"screen": "unconfirmed"}, shift=2.5, frames=6, zoom=laptop_box)
        if checks.get("screen"):
            checks["screen"]["zoom_box"] = laptop_box

    # Safe: a few seconds after it was first opened, is it still open with the cashier away from it?
    opened = [_num(p1.get("safe_opened_sec")), _num(p1.get("safe_left_open_after_use_sec"))]
    opened += [w["t"] for w in p2 if w["ans"].get("opens_safe") is True]
    opened = [t for t in opened if t is not None]
    if opened:
        t = min(opened)
        safe_box = ground(model, processor, path, box, vid, "safe",
                          "серый металлический шкаф (сейф) рядом со столом кассира", t + 4, dur)
        checks["safe"] = confirm(
            "safe", t + 3, t + 7, SAFE_PROMPT,
            lambda a: a.get("safe_door_open") is True and a.get("cashier_at_safe_working_with_it") is not True,
            {"safe_door_open": False}, zoom=safe_box)

    bag_rx = _num((p1.get("sealed_bag") or {}).get("received_sec"))
    bag_open = _num((p1.get("sealed_bag") or {}).get("opened_sec"))
    bag_safe = _num((p1.get("sealed_bag") or {}).get("cash_put_into_safe_sec"))
    bag_at_leave = (p1.get("at_leave") or {}).get("sealed_bag_on_desk") is True
    bag_ok = True
    if bag_rx is not None or bag_open is not None or bag_safe is not None or bag_at_leave:
        # Pass 1 claims a collection bag. Confirm with a neutral question first: a client handing over
        # banknotes is often reported as "a green bag", and the leading bag questions below agree with it.
        t = next(x for x in (bag_rx, bag_open, bag_safe, leave, 0.0) if x is not None)
        bp = confirm("bag_presence", t - 3, t + 4, BAG_PRESENCE_PROMPT, bag_present,
                     {"handed_item": "unconfirmed", "cloth_bag_on_desk": None}, shift=2.0)
        if bp:
            checks["bag_presence"] = bp
            bag_ok = bag_present(bp["ans"]) and not bp.get("_weak")

    if bag_ok and bag_rx is not None and bag_rx >= 2:   # handed over during the clip, not already on the desk
        checks["bag"] = confirm("bag", bag_rx - 3, bag_rx + 4, BAG_PROMPT,
                                lambda a: (a.get("green_bag_handed_to_cashier") is True
                                           and a.get("paper_or_document_handed_over") is False),
                                {"paper_or_document_handed_over": None}, shift=1.0)

    if bag_ok and (bag_open is not None or bag_safe is not None):
        t = bag_open if bag_open is not None else max(0.0, bag_safe - 8)
        checks["bag_open"] = confirm("bag_open", t - 2, t + 5, BAG_OPEN_PROMPT,
                                     lambda a: a.get("green_bag_opened_and_cash_taken_out") is True,
                                     {"green_bag_opened_and_cash_taken_out": None})

    coffee = _num(p1.get("coffee_drinking_with_cash_on_desk_sec"))
    if coffee is None:
        cups = [w["t"] for w in p2 if w["ans"].get("drinks_from_cup") is True]
        coffee = cups[0] if cups else None
    if coffee is not None:
        checks["coffee"] = confirm(
            "coffee", coffee - 3, coffee + 5, COFFEE_PROMPT,
            lambda a: a.get("cashier_drinks_from_cup") is True and a.get("banknotes_lying_on_desk") is True,
            {"banknotes_lying_on_desk": False})

    # Start of work: always look at the desk (personal items) and, if the cashier came in, the door.
    starts = _num(p1.get("cashier_starts_work_sec"))
    enters = _num(p1.get("cashier_enters_sec"))
    t_work = starts if starts is not None else (enters if enters is not None else 0.0)
    checks["personal"] = confirm(
        "personal", t_work + 1, t_work + 4, PERSONAL_PROMPT,
        lambda a: a.get("handbag_or_backpack_on_desk") is True or a.get("jacket_or_coat_on_desk") is True,
        {"handbag_or_backpack_on_desk": False, "jacket_or_coat_on_desk": False}, frames=6)
    if enters is not None:
        door_box = ground(model, processor, path, box, vid, "door",
                          "входная стеклянная дверь кассового помещения (створка с ручкой)", t_work + 2, dur, margin=0.3)
        if enters >= 1.5 and door_box:
            # She came in after the clip started: compare the door with how it looked before she entered.
            def compare(name, a, b):
                a, b = max(0.0, a), min(dur, b)
                if b - a < 0.8:
                    return None
                clip = side_by_side_clip(path, door_box, 0.2, a, b, os.path.join(CLIP_DIR, f"{vid}.{name}.mp4"))
                ans, raw = _ask_json(model, processor, clip, DOOR_COMPARE_PROMPT, 2.0, 6, 401_408, 200)
                return {"t": round(a, 1), "ans": ans or {}, **({} if ans else {"raw": raw[:300]})}
            first = compare("door_cmp", t_work + 1, t_work + 4)
            if first and first["ans"].get("door_open") is True and double_check:
                second = compare("door_cmp_2", t_work + 3, t_work + 6)
                first["second"] = second and second["ans"]
                if not (second and second["ans"].get("door_open") is True):
                    if recall:
                        first["_weak"] = True
                    else:
                        first["ans"] = {**first["ans"], "door_open": False, "_unconfirmed": True}
            checks["door"] = first
        else:
            checks["door"] = confirm("door", t_work + 1, t_work + 5, DOOR_PROMPT,
                                     lambda a: a.get("door_open") is True, {"door_open": False}, frames=6, zoom=door_box)

    arrivals = sorted(t for t in (_num(c.get("arrive_sec")) for c in (p1.get("clients") or [])
                                  if isinstance(c, dict)) if t is not None)

    # Document check before counting: from the client's arrival to just after counting starts.
    counting = sorted(w["t"] for w in p2 if w["ans"].get("counts_banknotes") is True)
    if counting:
        t_c = counting[0]
        t_a = arrivals[0] if arrivals and arrivals[0] < t_c else t_c - 8
        lo = max(t_a - 1, t_c - 15)
        checks["doc_check"] = confirm("doc", lo, t_c + 2, DOC_CHECK_PROMPT,
                                      lambda a: a.get("cashier_takes_and_reads_paper_from_client") is False,
                                      {"cashier_takes_and_reads_paper_from_client": None},
                                      shift=1.0, fps=2.0, frames=24)

    # Money "handed to the client": check the direction with a neutral question (deposits look the same).
    gives = [w for w in sorted(p2, key=lambda w: w["t"])
             if w["ans"].get("gives_banknotes_to_client") is True and not _before_counting(w, p2)]
    if gives:
        g = gives[0]["t"]
        tr = confirm("transfer", g - 2.5, g + 2.5, TRANSFER_PROMPT, returns_money,
                     {"money_moves": "unconfirmed"}, shift=1.5, fps=4.0, frames=20)
        if tr:
            checks["transfer"] = tr

    # Right after the last counting: authenticity check, or straight into a stack? (no client needed)
    counted = [w for w in p2 if w["ans"].get("counts_banknotes") is True]
    if counted:
        last = max(w["end"] for w in counted)
        ac = confirm("after_count", last - 1, min(dur, last + 7), AFTER_COUNT_PROMPT,
                     lambda a: after_count_action(a) == "stacks_or_bundles",
                     {"next_action": "unconfirmed"}, shift=1.5, fps=3.0, frames=20)
        if ac:
            checks["after_count"] = ac

    client_leaves = sorted(t for t in (_num(c.get("leave_sec")) for c in (p1.get("clients") or [])
                                       if isinstance(c, dict)) if t is not None)
    if client_leaves and counting:
        t_l = client_leaves[0]
        checks["no_client"] = confirm("noclient", t_l + 0.5, t_l + 5, NO_CLIENT_PROMPT,
                                      lambda a: a.get("client_at_desk") is False and a.get("cashier_counts_money") is True,
                                      {"cashier_counts_money": None}, shift=2.0, fps=2.0, frames=10)

    if len(arrivals) >= 2:
        # The second client's first ~20 s: handover of papers, where the first client's money is.
        span = min(20.0, dur - arrivals[1] + 1)
        checks["next_client"] = confirm(
            "next", arrivals[1] - 1, arrivals[1] - 1 + span, NEXT_CLIENT_PROMPT,
            lambda a: (a.get("money_of_two_clients_combined") is True
                       or (a.get("takes_paper_from_new_client") is True and a.get("previous_counting_still_ongoing") is True)
                       or a.get("previous_client_money_left_on_desk") is True),
            {"money_of_two_clients_combined": None, "previous_counting_still_ongoing": None,
             "previous_client_money_left_on_desk": None},
            shift=4.0, fps=1.2, frames=24, pixels=301_056)
    for name, zb in (("safe", locals().get("safe_box")), ("door", locals().get("door_box"))):
        if checks.get(name):
            checks[name]["zoom_box"] = zb
    return {k: c for k, c in checks.items() if c}


GROUND_PROMPT = """Найди на кадре банкнотосчётную машинку (счётчик купюр) на столе кассира.
Верни ТОЛЬКО JSON: {"bbox_2d": [x1, y1, x2, y2]} — координаты в диапазоне 0–1000 относительно кадра
(0,0 — левый верхний угол). Если машинки не видно, верни {"bbox_2d": null}."""

VERIFY_PROMPT = """Короткий фрагмент видео кассового помещения. Проверь, видно ли в нём следующее нарушение кассира:

«{definition}»

Верни ТОЛЬКО JSON: {{"what_happens": "<одно-два предложения>", "visible": <true|false>, "event_sec": <секунда внутри фрагмента, от 0>}}
Ставь "visible": true только если нарушение ясно видно в этом фрагменте. Если сомневаешься — false."""

SECOND_OPINION_PROMPT = os.path.join(HERE, "experiment", "v3_system.txt")


VISUAL_GUIDE = {
    "COUNTED_CASH_RETURNED": "Видно: кассир держит пачку купюр клиента и передаёт часть купюр обратно в руки клиенту.",
    "CASH_COUNTING_WITHOUT_DOCUMENT_CHECK": "Видно: клиент даёт деньги, кассир сразу начинает их пересчитывать, ни разу не взяв в руки лист/приглашение клиента.",
    "PREVIOUS_CLIENT_CASH_NOT_SECURED": "Видно: на столе лежит пачка денег первого клиента, а кассир уже принимает деньги/документ следующего клиента.",
    "CLIENTS_CASH_MIXED": "Видно: кассир кладёт деньги второго клиента прямо на стопку денег первого клиента.",
    "NEXT_CLIENT_STARTED_BEFORE_PREVIOUS_COMPLETED": "Видно: кассир ещё считает деньги первого клиента и одновременно берёт лист у второго клиента.",
    "CASH_COUNTED_WITHOUT_CLIENT_PRESENT": "Видно: клиент встаёт и уходит из помещения, а кассир продолжает пересчитывать его деньги.",
    "CASH_AUTHENTICITY_NOT_CHECKED": "Видно: после пересчёта кассир сразу складывает купюры в одну стопку, не поднося их к детектору/лампе.",
}


def violation_definitions(prompt_path=SECOND_OPINION_PROMPT):
    """TYPE -> definition text, parsed from the numbered categories of the system prompt."""
    import re
    try:
        text = open(prompt_path, encoding="utf-8").read()
    except OSError:
        return {}
    out = {}
    for m in re.finditer(r"^\d+\. ([A-Z_]+)\n(.*?)(?=^\d+\. [A-Z_]+\n|^=====)", text, re.M | re.S):
        out[m.group(1)] = " ".join(m.group(2).split())
    return out


def machine_check(model, processor, path, box, vid, p1, p2, dur):
    """Was the banknote counter busy between 'bag opened' and 'cash into safe'? Locates the counter
    once with the model (bounding box), then measures motion inside it without the model."""
    bag = (p1 or {}).get("sealed_bag") or {}
    lo = _num(bag.get("opened_sec"))
    lo = lo if lo is not None else _num(bag.get("received_sec"))
    hi = _num(bag.get("cash_put_into_safe_sec"))
    if lo is None or hi is None or hi - lo < 2:
        return None
    clip = cut_clip(path, box, max(0.0, lo - 1), min(dur, lo + 1), os.path.join(CLIP_DIR, f"{vid}.ground.mp4"),
                    target_pixels=401_408)
    ans, raw = _ask_json(model, processor, clip, GROUND_PROMPT, 1.0, 2, 401_408, 80)
    bb = (ans or {}).get("bbox_2d")
    if not (isinstance(bb, list) and len(bb) == 4 and all(isinstance(x, (int, float)) for x in bb)):
        return {"t": lo, "ans": {"machine_running": None, "bbox": None}, "raw": (raw or "")[:200]}
    bx0, by0, bx1, by1 = box
    bw, bh = bx1 - bx0, by1 - by0
    x0, y0, x1, y1 = [max(0, min(1000, v)) / 1000 for v in bb]
    # Inner 60% of the box: the bill path, less sensitive to hands passing over the edges.
    cx0, cx1 = x0 + 0.2 * (x1 - x0), x1 - 0.2 * (x1 - x0)
    cy0, cy1 = y0 + 0.2 * (y1 - y0), y1 - 0.2 * (y1 - y0)
    mbox = (int(bx0 + cx0 * bw), int(by0 + cy0 * bh), int(bx0 + cx1 * bw), int(by0 + cy1 * bh))
    if mbox[2] - mbox[0] < 6 or mbox[3] - mbox[1] < 6:
        return {"t": lo, "ans": {"machine_running": None, "bbox": mbox}}
    active = 0.0
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or 30)
        step = max(1, round(fps / 4))
        prev = None
        for i, fr in enumerate(c.decode(s)):
            t = i / fps
            if t < lo or i % step:
                continue
            if t > hi:
                break
            g = np.asarray(fr.to_image().convert("L").crop(mbox), np.float32)
            if prev is not None:
                d = g - prev
                if (np.abs(d - np.median(d)) > 15).mean() > 0.08:
                    active += step / fps
            prev = g
    return {"t": lo, "ans": {"machine_running": active >= 1.5, "active_sec": round(active, 2), "bbox": mbox}}


def second_opinion(model, processor, path, box, vid, p1, dur, found_types, double_check=True, recall=True):
    """Run the original single-pass prompt; every type it claims that the rules did not already find
    becomes a candidate that must be confirmed by focused, double-checked clips."""
    if not os.path.exists(SECOND_OPINION_PROMPT):
        return {}, None
    system = open(SECOND_OPINION_PROMPT, encoding="utf-8").read().strip()
    overview = os.path.join(CLIP_DIR, f"{vid}.overview.mp4")
    answer, _ = infer_video.ask_video(model, processor, overview, infer_video.DEFAULT_PROMPT_WITH_SYSTEM, system,
                                      1.0, 32, 150_528, 600, repetition_penalty=1.05)
    parsed = infer_video.extract_json(answer) or {}
    defs = violation_definitions()
    leave = _num((p1 or {}).get("cashier_leaves_sec"))
    out = {}
    for vio in parsed.get("violations") or []:
        typ = vio.get("type") if isinstance(vio, dict) else None
        if typ not in defs or typ not in VISUAL_GUIDE or typ in found_types or typ in out:
            continue   # types with a dedicated check (door, safe, leave state, bag, ...) are decided by it
        t = _num(vio.get("timecode_sec"))
        if (t is None or t < 1) and typ.endswith(("LEAVING", "NOT_SECURED", "UNATTENDED", "IN_LOCK")) and leave is not None:
            t = leave
        if t is not None and 1 <= t < dur:
            spans = [(t - 4, t + 6), (t - 2, t + 8)]
            kw = {"fps": 2.0, "frames": 20}
        else:                                       # no usable time: look at the whole clip, sparsely
            spans = [(0, dur), (0.5, dur)]
            kw = {"fps": 1.0, "frames": 24}
        prompt = VERIFY_PROMPT.format(definition=defs[typ] + " " + VISUAL_GUIDE.get(typ, ""))
        first = _check(model, processor, path, box, vid, f"cand_{typ.lower()}", *spans[0], prompt, dur, **kw)
        if not first:
            continue
        first["description"] = vio.get("description") or ""
        if first["ans"].get("visible") is True and double_check:
            second = _check(model, processor, path, box, vid, f"cand_{typ.lower()}_2", *spans[1], prompt, dur, **kw)
            first["second"] = second and second["ans"]
            if not (second and second["ans"].get("visible") is True):
                if recall:
                    first["_weak"] = True
                else:
                    first["ans"] = {**first["ans"], "visible": False, "_unconfirmed": True}
        out[typ] = first
    return out, parsed


CASH_WORDS = ("купюр", "банкнот", "деньг", "наличн", "banknote", "money", "cash", "bill")
SMALL_PAPER = (("small", "paper"), ("small", "note"), ("маленьк", "бумаг"), ("небольш", "бумаг"),
               ("маленьк", "лист"), ("небольш", "лист"))
STAMP_WORDS = ("штамп", "печат", "цилиндр", "stamp", "cylind", "бланк", "документ", "document", "папк", "folder")


def _leave_state(model, processor, path, box, vid, p1, dur, double_check=True, recall=True):
    """Snapshot of the desk/safe/screen after the cashier leaves. Falls back to the last seconds of
    the clip when pass 1 did not report the cashier leaving. Returns (p1, snapshot_debug)."""
    leave_t = _num(p1.get("cashier_leaves_sec"))
    snap, second, source = None, None, None

    def look(name, t0, t1, prompt):
        t0, t1 = max(0.0, t0), min(dur, t1)
        if t1 - t0 < 0.8:
            return None
        clip = cut_clip(path, box, t0, t1, os.path.join(CLIP_DIR, f"{vid}.{name}.mp4"), target_pixels=400_000)
        return _ask_json(model, processor, clip, prompt, 2.0, 8, 401_408, 300)[0]

    if leave_t is not None and leave_t + 0.5 < dur:
        snap, source = look("leave", leave_t + 0.5, leave_t + 4.0, LEAVE_PROMPT), "pass1"
        if snap and double_check and leave_t + 4.5 < dur:
            second = look("leave_2", leave_t + 4.0, leave_t + 7.5, LEAVE_PROMPT)
    else:
        # Safety net: is the cashier still at the desk at the end of the clip?
        end = look("end", dur - 3.5, dur, END_PROMPT)
        if end and end.get("cashier_present_at_desk") is False:
            snap, source = end, "end_of_clip"
            leave_t = last_motion_time(path, box)
            leave_t = leave_t if leave_t is not None else max(0.0, dur - 4.0)
            if double_check:
                second = look("end_2", dur - 6.5, dur - 3.0, END_PROMPT)
            p1 = {**p1, "cashier_leaves_sec": round(leave_t, 1)}

    if snap and source == "pass1":
        seen_there = [x.get("cashier_at_workplace") is True for x in (snap, second) if x]
        if seen_there and all(seen_there):
            # Pass 1 probably mistook a client (or someone else) leaving for the cashier leaving.
            p1 = {**p1, "cashier_leaves_sec": None, "_leave_rejected": leave_t}
            return p1, {"source": "rejected: cashier still at workplace", "first": snap, "second": second}

    if snap:
        state = {k: v for k, v in snap.items() if isinstance(v, bool) and k != "cashier_at_workplace"}
        weak = []
        # The model often lists stamps without naming them ("two cylindrical containers"); in recall
        # mode a matching desk item counts as a weak stamps/documents finding.
        items = snap.get("desk_items")
        items = " ".join(map(str, items)) if isinstance(items, list) else str(items or "")
        if (recall and state.get("stamps_or_documents_on_desk") is not True
                and any(k in items.lower() for k in STAMP_WORDS)):
            state["stamps_or_documents_on_desk"] = True
            weak.append("stamps_or_documents_on_desk")
        low = items.lower()
        if recall and state.get("cash_on_desk") is not True and (
                any(k in low for k in CASH_WORDS)
                or any(a in low and b in low for a, b in SMALL_PAPER)):
            state["cash_on_desk"] = True
            weak.append("cash_on_desk")
        if second:   # a positive the second look does not repeat is weak (recall) or dropped
            s2 = {k: v for k, v in second.items() if isinstance(v, bool)}
            weak = sorted(set(weak) | {k for k, v in state.items() if v is True and s2.get(k) is not True})
            if not recall:
                state = {k: (False if k in weak else v) for k, v in state.items()}
                weak = []
        p1 = {**p1, "at_leave": {**(p1.get("at_leave") or {}), **state}, "_leave_double": bool(second),
              "_leave_weak": weak}
    return p1, {"source": source, "first": snap, "second": second}


def _analyze(model, processor, path, cfg):
    vid = os.path.splitext(os.path.basename(path))[0]
    box, _ = find_action_box(path)
    with av.open(path) as c:
        dur = float(c.duration / 1e6)
    # Longer clips get more overview frames and more zoomed windows.
    p1_frames = cfg["p1_frames"] if dur <= 45 else 48
    n_windows = int(min(6, max(cfg["windows"], dur // 12)))

    os.makedirs(CLIP_DIR, exist_ok=True)
    _progress("overview")
    overview = cut_clip(path, box, 0, 1e9, os.path.join(CLIP_DIR, f"{vid}.overview.mp4"),
                        target_pixels=(box[2] - box[0]) * (box[3] - box[1]), max_fps=4.0)
    p1_pixels = cfg["p1_pixels"]
    if time_left() < 70:          # several videos queued: a lighter overview keeps everyone within the limit
        p1_frames, p1_pixels = min(p1_frames, 20), min(p1_pixels, 100_352)
    p1, raw1 = _ask_json(model, processor, overview, PASS1_PROMPT, cfg["p1_fps"], p1_frames,
                         p1_pixels, 900)
    p1 = p1 or {}

    if time_left() < 90:
        n_windows = min(n_windows, 2 if time_left() > 45 else 1 if time_left() > 30 else 0)
    p2 = []
    _progress("detail")
    windows, _ = find_windows(path, box=box, max_windows=n_windows)
    for k, w in enumerate(windows):
        clip = cut_clip(path, w["zoom_box"], w["start"], w["end"], os.path.join(CLIP_DIR, f"{vid}.w{k}.mp4"))
        ans, raw = _ask_json(model, processor, clip, PASS2_PROMPT.format(start=w["start"], end=w["end"]),
                             cfg["p2_fps"], cfg["p2_frames"], cfg["p2_pixels"], 300)
        ans = ans or {}
        rel = _num(ans.get("event_sec"))
        t = w["start"] + rel if rel is not None and 0 <= rel <= (w["end"] - w["start"]) else w["peak"]
        p2.append({"start": w["start"], "end": w["end"], "t": round(t, 1), "ans": ans, "raw": raw[:300]})

    double = cfg["double_check"] and time_left() > 100
    _progress("leave_check")
    p1, leave_dbg = _leave_state(model, processor, path, box, vid, p1, dur, double, cfg["recall_mode"])
    _progress("focused_checks")
    checks = run_checks(model, processor, path, box, vid, p1, p2, dur, double, cfg["recall_mode"])
    mc = machine_check(model, processor, path, box, vid, p1, p2, dur) if time_left() > 60 else None
    if mc:
        checks["machine"] = mc
    all_violations = apply_rules(p1, p2, checks)
    opinion = None
    if cfg.get("second_opinion", True) and time_left() > 75:
        _progress("audit")
        cands, opinion = second_opinion(model, processor, path, box, vid, p1, dur,
                                        {x["type"] for x in all_violations}, cfg["double_check"], cfg["recall_mode"])
        if cands:
            checks["candidates"] = cands
            all_violations = apply_rules(p1, p2, checks)
    violations = all_violations
    if cfg["single_violation"] and len(violations) > 1:
        rank = {t: i for i, t in enumerate(PRIORITY)}
        # Confirmed beats weak; among confirmed, scenario-specific types beat generic ones (a double-
        # confirmed door/personal-items finding must not hide a real bag/client violation).
        violations = [min(violations, key=lambda x: (x.get("_ev", 1) == 0, rank.get(x["type"], len(PRIORITY)),
                                                     -x.get("_ev", 1)))]
    violations = [{k: x[k] for k in ("type", "timecode_sec", "description")} for x in violations]
    result = {"verdict": "violation" if violations else "no_violation", "violations": violations}
    debug = {"box": box, "dur": dur, "pass1": p1, "pass1_raw": raw1 if not p1 else None, "leave": leave_dbg,
             "checks": checks, "all_violations": all_violations, "second_opinion": opinion,
             "pass2": [{k: v for k, v in w.items() if k != "raw" or not w["ans"]} for w in p2]}
    return result, debug


def analyze(model, processor, path, cfg=None):
    """Never raises: on any failure returns a valid no_violation answer and the error in debug."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    try:
        return _analyze(model, processor, path, cfg)
    except Exception as exc:  # noqa: BLE001 - the output must stay valid JSON for every video
        import traceback
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return ({"verdict": "no_violation", "violations": []},
                {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]})


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--model-dir", default=infer_video.MODEL_DIR)
    ap.add_argument("--precision", choices=["4bit", "8bit", "bf16"], default="4bit")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    model, processor = infer_video.load_model(args.model_dir, args.precision)
    result, debug = analyze(model, processor, args.video)
    if args.debug:
        print(json.dumps(debug, ensure_ascii=False, indent=2), file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
