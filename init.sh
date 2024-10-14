#!/bin/bash
cd whope_backend
poetry run daphne whope_backend.asgi:application
