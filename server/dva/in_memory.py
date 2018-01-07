from django.conf import settings  # noqa
import redis

redis_client = redis.Redis(host=settings.REDIS_HOST,port=settings.REDIS_PORT,password=settings.REDIS_PASSWORD)
