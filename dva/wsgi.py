"""
WSGI config for dva project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.10/howto/deployment/wsgi/
"""

import os

if 'VDN_ONLY_MODE' in os.environ:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    from django.core.wsgi import get_wsgi_application
    from whitenoise.django import DjangoWhiteNoise
    application = get_wsgi_application()
    application = DjangoWhiteNoise(application)
else:
    from django.core.wsgi import get_wsgi_application
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    application = get_wsgi_application()
