import os
import uvicorn
from app.api import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
