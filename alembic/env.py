"""Alembic environment.

The database URL comes from `scribe.config.Settings`, not `alembic.ini`, so
migrations, the API, and workers can never disagree about which database they
are pointing at — in k8s the URL arrives as an env var and nothing else needs
changing.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from scribe.config import load_settings
from scribe.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Fall back to the application's configured database, but never override a
# URL the caller already set. Overriding unconditionally means a caller that
# explicitly targets another database -- a test using a throwaway one, or an
# operator migrating a staging copy -- is silently redirected at the real one.
# `downgrade base` then drops every table in it. This has already destroyed
# data once.
if not config.get_main_option("sqlalchemy.url", None):
    config.set_main_option("sqlalchemy.url", load_settings().database_url)


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
