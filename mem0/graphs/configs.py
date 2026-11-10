from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class ArcadeDBConfig(BaseModel):
    """Connection settings. The database must already exist; the provider only creates its own schema."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(description="ArcadeDB HTTP URL, e.g. http://localhost:2480")
    database: str = Field(description="Name of an existing database")
    username: str = Field(description="ArcadeDB username")
    password: SecretStr = Field(description="ArcadeDB password")
    namespace: Optional[str] = Field(
        default=None,
        description="Graph namespace. Defaults to the vector store collection name; set it explicitly when "
        "separate vector datasets share a collection name.",
    )
    timeout: float = Field(default=30.0, gt=0, description="HTTP timeout in seconds")


class GraphStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["arcadedb"] = Field(default="arcadedb", description="Graph store provider")
    config: ArcadeDBConfig = Field(description="Provider configuration")
