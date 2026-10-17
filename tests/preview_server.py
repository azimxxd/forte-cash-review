"""TEST ONLY visual QA host. Isolated empty storage, disabled model, no production secrets.
The harness supplies its own test-only UI credentials to exercise the built frontend.
Never use this as a deployment entry point. Binds loopback only.
"""

import base64
import tempfile
from pathlib import Path
import uvicorn
from app.config import Settings
from app.main import create_app

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="cash-ui-test-") as root:
        settings = Settings(
            _env_file=None,
            api_key="test-api-only-000000000000",
            ui_password="test-ui-only-0000000000000",
            data_dir=Path(root),
            model_enabled=False,
        )
        app = create_app(settings)
        token = base64.b64encode(f"{settings.ui_user}:{settings.ui_password}".encode())

        async def harness(scope, receive, send):
            if scope["type"] == "http":
                scope = dict(scope)
                scope["headers"] = [
                    (k, v) for k, v in scope["headers"] if k != b"authorization"
                ] + [(b"authorization", b"Basic " + token)]
            await app(scope, receive, send)

        uvicorn.run(harness, host="127.0.0.1", port=8010, access_log=False)
