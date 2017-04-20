from django.conf.urls import url,include
import views
from django.contrib.auth import views as auth_views
from rest_framework.authtoken import views as auth_token_views



urlpatterns = [
    url(r'^login/$', auth_views.login, name='login'),
    url(r'^logout/$', auth_views.logout, name='logout'),
    url(r'^accounts/login/$', auth_views.login, name='login'),
    url(r'^accounts/logout/$', auth_views.logout, name='logout'),
    url(r'^password_reset/$', auth_views.password_reset, name='password_reset'),
    url(r'^$', views.marketing, name='marketing'),
    url(r'^get_token/$', views.get_token, name='token'),
    url(r'^accounts/profile/$', views.marketing, name='marketing'),
    url(r'^api/', include(views.router.urls)),
    url(r'^api-token-auth/', auth_token_views.obtain_auth_token)
]
