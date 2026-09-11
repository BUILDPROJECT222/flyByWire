# Data (local)

MaleCNS downloads and prepared graphs live in `malecns/`.

Most files are gitignored. After clone:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m lab.download
.venv/bin/python -m lab.prepare
.venv/bin/python -m lab.prepare_full
.venv/bin/python -m lab.full_vision
```
