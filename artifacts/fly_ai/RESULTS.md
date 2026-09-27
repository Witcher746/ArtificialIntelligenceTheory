# Проверенный результат

## Race — подтверждённый pixel-only baseline

Фиксированный запуск: seed 42, 12 teacher episodes, 0 TD episodes,
20 невиданных evaluation seed, максимум 300 решений на эпизод.

| Policy | Средний счёт | Медианный счёт | Среднее число решений |
|---|---:|---:|---:|
| Случайная | 129,11 | 130,27 | 81,20 |
| Обученная CV-policy | **349,59** | **312,75** | **138,45** |

Средний счёт вырос в 2,71 раза (`+170,8%`), а средняя длительность эпизода —
на `70,5%`. Train seed и evaluation seed не пересекаются. Во время evaluation
policy получает только RGB-кадры; teacher и игровые координаты отключены.
В попарном сравнении на одинаковых seed policy выиграла 16 из 20 эпизодов;
средняя разница счёта `+220,48`, bootstrap 95% CI `[+132,63; +309,79]`.
Контрольная TD-донастройка на 5 эпизодах дала 343,97 и не улучшила 349,59,
поэтому в canonical checkpoint оставлена более сильная BC-policy.

Артефакты:

- `race_fly_agent.npz` — checkpoint;
- `race_episodes.csv` — все train/random/agent эпизоды;
- `race_report.json` — агрегированные числа.

SHA-256 checkpoint:
`96246614c70c6efc4fc989d9266670f89e1d060e74c68b45f7d832abc8a548e8`.

Проверка агрегатов и парного bootstrap CI (50 000 повторов, seed 2026):

```powershell
.\.venv\Scripts\python.exe -m fly_ai.analyze_results artifacts\fly_ai\race_episodes.csv
```

Команда воспроизведения:

```powershell
.\.venv\Scripts\python.exe -m fly_ai.train `
  --game race --teacher-episodes 12 --rl-episodes 0 `
  --eval-episodes 20 --max-steps 300 --kenyon 4096 `
  --learning-rate 0.015 --seed 42 --output-dir artifacts\fly_ai
```

## Остальные игры

Все шесть адаптеров прошли headless smoke tests. Для Knife Hit был проведён
исследовательский прогон, но средний score обученной policy не превысил сильный
случайный baseline устойчиво. Поэтому Knife Hit и ещё четыре сложные среды не
помечены как «решённые» и не должны использоваться в презентации как готовый
результат.
