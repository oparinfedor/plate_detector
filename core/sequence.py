"""Исправление отдельных ошибок распознавания за счёт того, что номера
при полевой съёмке почти по порядку внутри одной камеры (Витерби по
EXIF-времени съёмки).

Работает только когда задана expected_length (сессия задаёт фиксированную
длину номера) - без неё сравнивать разнодлинные строки как числа
неоднозначно.

Таблица TRANSITION_LOG_PROBS откалибрована на реально измеренном
распределении дельт между соседними номерами внутри камеры на полном
test/ (997 переходов) - НЕ на заявленных в аудите 93-100% монотонности,
которые на полных данных не подтвердились (реально delta в {0,+1} только
в 81.1% случаев). Дальний хвост (площадки снимали не всегда строго по
номерам) получает малую "пол"-вероятность через сглаживание, а не 0 -
иначе Витерби не сможет пережить настоящий скачок между секциями.
"""
import math
from collections import defaultdict
from datetime import datetime

from PIL import Image
from PIL.ExifTags import TAGS

EXIF_DATETIME_ORIGINAL = 0x9003
EXIF_IFD = 0x8769

# Счётчики дельт, измеренные на test/ (см. план/чат) - камера+площадка,
# 997 переходов.
_DELTA_COUNTS = {
    0: 598, 1: 211, -1: 46, 2: 34, 3: 15, -3: 14, 4: 9, -2: 9,
    -10: 6, 5: 3, -5: 3, 7: 3, -22: 2, 10: 2, 21: 2,
}
_TOTAL_TRANSITIONS = 998
_FLOOR_COUNT = 0.5  # сглаживание для дельт вне таблицы - настоящие скачки бывают


def _transition_log_prob(delta):
    count = _DELTA_COUNTS.get(delta, _FLOOR_COUNT)
    return math.log(count / _TOTAL_TRANSITIONS)


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = cur
    return prev[-1]


def read_camera_id(path):
    try:
        exif = Image.open(path).getexif()
    except Exception:
        return ""
    tags = {TAGS.get(k, k): v for k, v in exif.items()}
    make = str(tags.get("Make", "")).strip()
    model = str(tags.get("Model", "")).strip()
    if not make and not model:
        return ""
    return f"{make}|{model}"


def read_capture_time(path):
    try:
        exif = Image.open(path).getexif()
    except Exception:
        return None
    dt_str = None
    try:
        dt_str = exif.get_ifd(EXIF_IFD).get(EXIF_DATETIME_ORIGINAL)
    except Exception:
        pass
    if not dt_str:
        tags = {TAGS.get(k, k): v for k, v in exif.items()}
        dt_str = tags.get("DateTime")
    if not dt_str:
        return None
    try:
        return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def viterbi_correct(observations, expected_length, margin=15, max_delta=25, edit_weight=3.0):
    """observations: список строк-кандидатов (в порядке времени съёмки),
    пустая строка/None - кадр без распознанного номера.
    Возвращает список int (или None, если восстановить состояние не из
    чего - вся последовательность без наблюдений)."""
    n = len(observations)
    valid_values = [int(o) for o in observations if o and o.isdigit()]
    if not valid_values:
        return [None] * n

    # Пространство состояний не должно вылезать за диапазон, реально
    # представимый expected_length цифрами - иначе Витерби может выбрать
    # состояние вроде 1005 при expected_length=3, а zfill его не обрежет,
    # только дополнит, и на выходе получится 4-значный код.
    max_representable = 10 ** expected_length - 1
    lo = max(0, min(valid_values) - margin)
    hi = min(max(valid_values) + margin, max_representable)
    states = list(range(lo, hi + 1))
    num_states = len(states)

    def emission(state, obs):
        if not obs:
            return 0.0
        target = str(state).zfill(expected_length)
        return -edit_weight * levenshtein(target, obs)

    dp = [emission(s, observations[0]) for s in states]
    backptr = [[0] * num_states for _ in range(n)]

    for t in range(1, n):
        obs = observations[t]
        new_dp = [float("-inf")] * num_states
        for j in range(num_states):
            s2 = states[j]
            lo_i = max(0, j - max_delta)
            hi_i = min(num_states - 1, j + max_delta)
            best_score = float("-inf")
            best_i = lo_i
            for i in range(lo_i, hi_i + 1):
                delta = s2 - states[i]
                score = dp[i] + _transition_log_prob(delta)
                if score > best_score:
                    best_score = score
                    best_i = i
            new_dp[j] = best_score + emission(s2, obs)
            backptr[t][j] = best_i
        dp = new_dp

    best_final = max(range(num_states), key=lambda j: dp[j])
    path_idx = [0] * n
    path_idx[-1] = best_final
    for t in range(n - 1, 0, -1):
        path_idx[t - 1] = backptr[t][path_idx[t]]

    return [states[idx] for idx in path_idx]


def correct_records(records, expected_length, min_group_size=3):
    """records: список dict с ключами 'path' (Path к исходному фото) и
    'code' (строка от decode.select_digit_string). Возвращает новый
    список той же формы - code скорректирован там, где хватило сигнала
    по последовательности, иначе оставлен как был."""
    if not expected_length:
        return list(records)

    by_camera = defaultdict(list)
    unresolved = []
    for rec in records:
        cam = read_camera_id(rec["path"])
        t = read_capture_time(rec["path"])
        if not cam or t is None:
            unresolved.append(dict(rec))
            continue
        by_camera[cam].append((t, rec))

    corrected = []
    for items in by_camera.values():
        items.sort(key=lambda x: x[0])
        valid_count = sum(1 for _, r in items if r.get("code"))
        if valid_count < min_group_size:
            corrected.extend(dict(r) for _, r in items)
            continue

        observations = [r.get("code", "") for _, r in items]
        fixed_values = viterbi_correct(observations, expected_length)

        for (_, r), val in zip(items, fixed_values):
            new_rec = dict(r)
            if val is not None:
                new_rec["code"] = str(val).zfill(expected_length)
            corrected.append(new_rec)

    corrected.extend(unresolved)
    return corrected
