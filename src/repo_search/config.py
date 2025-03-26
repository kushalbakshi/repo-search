"""Configuration management for RepoSearch."""

import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv


class Config:
    """Configuration for RepoSearch.

    Handles loading environment variables and providing configuration values.
    """

    def __init__(self, env_file: Optional[str] = None) -> None:
        """Initialize the configuration.

        Args:
            env_file: Path to the .env file. If None, will look for .env in the current
                working directory.
        """
        # Try to load the environment variables from a .env file
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        # API keys
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

        # GitHub settings
        self.github_token: Optional[str] = os.getenv("GITHUB_TOKEN")

        # Storage settings
        self.data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
        self.db_path: Path = self.data_dir / "db"

        # Ensure data directories exist
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.db_path.mkdir(exist_ok=True, parents=True)

        # Embedding settings
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

        # Chunking settings
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

        # Search settings
        self.max_results: int = int(os.getenv("MAX_RESULTS", "10"))
        self.score_threshold: float = float(os.getenv("SCORE_THRESHOLD", "0.0"))

    def to_dict(self) -> Dict[str, str]:
        """Convert the configuration to a dictionary for serialization.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "gemini_api_key": "***" if self.gemini_api_key else "",
            "openai_api_key": "***" if self.openai_api_key else "",
            "github_token": "***" if self.github_token else None,
            "data_dir": str(self.data_dir),
            "db_path": str(self.db_path),
            "embedding_model": self.embedding_model,
            "embedding_batch_size": str(self.embedding_batch_size),
            "chunk_size": str(self.chunk_size),
            "chunk_overlap": str(self.chunk_overlap),
            "max_results": str(self.max_results),
            "score_threshold": str(self.score_threshold),
        }


# Create a global instance of the configuration
config = Config()
