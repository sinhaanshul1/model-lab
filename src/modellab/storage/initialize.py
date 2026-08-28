"""Create ModelLab's local PostgreSQL tables."""

from modellab.storage.database import create_schema


if __name__ == "__main__":
    create_schema()
    print("ModelLab PostgreSQL schema is ready.")
