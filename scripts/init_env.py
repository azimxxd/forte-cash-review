"""Create local secrets once, never print them. No credentials are put in frontend assets."""

import os
import secrets
from pathlib import Path

os.umask(0o077)
path = Path(".env")
content = Path(".env.example").read_text()
content = content.replace(
    "CASH_API_KEY=\n", "CASH_API_KEY=" + secrets.token_urlsafe(36) + "\n"
)
content = content.replace(
    "CASH_UI_PASSWORD=\n", "CASH_UI_PASSWORD=" + secrets.token_urlsafe(36) + "\n"
)
with path.open("x") as f:
    f.write(content)
print("Created .env with mode 0600. Existing secrets are never overwritten.")
