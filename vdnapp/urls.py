from django.conf.urls import url,include
import views
from rest_framework.authtoken import views as auth_token_views



urlpatterns = [
    url(r'^$', views.marketing, name='marketing'),
    url(r'^get_token/$', views.get_token, name='token'),
    url(r'^accounts/profile/$', views.marketing, name='marketing'),
    url(r'^api/', include(views.router.urls)),
    url(r'^api-token-auth/', auth_token_views.obtain_auth_token)
]
