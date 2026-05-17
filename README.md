# Finding promising stocks
- prediction backends (selectable via env `PREDICTER`):
  - `garch` (default) — GARCH(1,1) on log returns; drift from historical sample mean
  - `prophet` — Facebook Prophet with optional US-market holidays and weekly/yearly seasonality

Swap by setting `PREDICTER=prophet` (or `garch`) in the environment / `.env`.

<img width="395" alt="Screenshot 2024-08-08 at 23 40 04" src="https://github.com/user-attachments/assets/69133dfa-567b-4643-9492-77a044e102c7">
