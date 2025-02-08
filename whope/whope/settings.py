"""
Django settings for whope project.

Generated by "django-admin startproject" using Django 5.1.2.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.1/ref/settings/
"""

import os
from datetime import timedelta
from logging import ERROR, Logger, getLogger
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)
from pymongo import ASCENDING
from utils.crypto_utils import generate_asymmetric_keys
import string
import random
from tensorflow.keras.models import load_model
from transformers import BertTokenizer

load_dotenv()

# Build paths inside the project like this: BASE_DIR / "subdir".
BASE_DIR: Path = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY: str = os.getenv("SECRET_KEY")

# SECURITY WARNING: don"t run with debug turned on in production!
DEBUG: bool = True

ALLOWED_HOSTS: List[str] = []


# Application definition

INSTALLED_APPS: List[str] = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "channels",
    "register",
    "login",
    "chat",
]

MIDDLEWARE: List[str] = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF: str = "whope.urls"

TEMPLATES: List[Dict[str, Any]] = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION: str = "whope.wsgi.application"


# Database
# https://docs.djangoproject.com/en/5.1/ref/settings/#databases

DATABASES: Dict[str, Dict[str, Any]] = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS: List[Dict[str, str]] = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.1/topics/i18n/

LANGUAGE_CODE: str = "en-us"

TIME_ZONE: str = "UTC"

USE_I18N: bool = True

USE_TZ: bool = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.1/howto/static-files/

STATIC_URL: str = "static/"

# Default primary key field type
# https://docs.djangoproject.com/en/5.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD: str = "django.db.models.BigAutoField"

ASGI_APPLICATION: str = "whope.asgi.application"

# HF_TOKEN: str = os.getenv("HF_TOKEN")

CHANNEL_LAYERS: Dict[str, Dict[str, Any]] = {"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}

MONGODB_URI: str = os.getenv("MONGODB_URI")
RABBITMQ_URI: str = os.getenv("RABBITMQ_URI")
CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND")

MONGODB_CLIENT: AsyncIOMotorClient = AsyncIOMotorClient(MONGODB_URI)
# this calling of the database is just done to create the indexes on startup, but then when the database is needed,
# it will be called with get_motor_db to ensure that its async loop is the same as the one of the method that calls it
DATABASE: AsyncIOMotorDatabase = MONGODB_CLIENT["whope"]
# DATABASE_TEST: AsyncIOMotorDatabase = MONGODB_CLIENT["whope_test"]
USERS: AsyncIOMotorCollection = DATABASE["users"]
USERS.create_index([("username", ASCENDING)], unique=True)
USERS.create_index([("is_worker", ASCENDING), ("status", ASCENDING)])

async def get_motor_db() -> AsyncIOMotorDatabase:
    client: AsyncIOMotorClient = AsyncIOMotorClient(MONGODB_URI)
    db: AsyncIOMotorDatabase = client["whope"]
    return db

pika_logger: Logger = getLogger("aio_pika")
pika_logger.setLevel(ERROR)

PRIVATE_KEY_PASSWORD: bytes = bytes("".join(random.choices(string.ascii_letters, k=32)), "utf-8")

PRIVATE_KEY_BYTES, PUBLIC_KEY_BYTES = generate_asymmetric_keys(PRIVATE_KEY_PASSWORD)

CELERY_TASK_IGNORE_RESULT = True
CELERY_TASK_RESULT_EXPIRES = 0

REST_FRAMEWORK: Dict[str, tuple] = {
    "DEFAULT_AUTHENTICATION_CLASSES": ("rest_framework_simplejwt.authentication.JWTAuthentication",),
}

SIMPLE_JWT: Dict[str, timedelta] = {
    "ACCESS_TOKEN_LIFETIME": timedelta(days=300),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=1),
}

TOPIC_MODEL_PATH = os.path.join(BASE_DIR, 'models/topics_dense_model.keras')
TOPIC_MODEL = load_model(TOPIC_MODEL_PATH)

EMOTION_MODEL_PATH = os.path.join(BASE_DIR, 'models/emotions_dense_model.keras')
EMOTION_MODEL = load_model(EMOTION_MODEL_PATH)

TOKENIZER: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

HF_TOKEN: str = os.getenv("HF_TOKEN")

# TODO: delete this
# import tensorflow as tf
# print("Build info", tf.sysconfig.get_build_info())
# print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

NLM_RULES_PATH: Path = os.path.join(BASE_DIR, 'utils/nlm_rules.pip')
