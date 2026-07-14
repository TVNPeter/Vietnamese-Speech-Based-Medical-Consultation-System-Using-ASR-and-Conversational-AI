"""Application entry point."""

import uvicorn

from src.config import settings


def main() -> None:
    """Start the FastAPI server."""
    uvicorn.run(
        "src.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )


if __name__ == "__main__":
    main()
