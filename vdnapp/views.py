from django.shortcuts import render,get_object_or_404
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required,user_passes_test
from django.core.exceptions import MultipleObjectsReturned
from django.shortcuts import redirect
from django.views.generic import ListView,DetailView
from django.utils.decorators import method_decorator
from .models import Dataset,User,Organization,Annotation,VDNRemoteDetector
from collections import defaultdict
from django.template.defaulttags import register
import google
import os,logging
from rest_framework import viewsets,mixins
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from rest_framework import serializers, viewsets
from django.contrib.auth.models import User
from rest_framework import routers
from rest_framework.authtoken.models import Token


class DatasetSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        depth=1
        model = Dataset
        fields = '__all__'


class VDNRemoteDetectorSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        depth=1
        model = VDNRemoteDetector
        fields = '__all__'


class AnnotationSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Annotation
        fields = '__all__'


class OrganizationSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Organization
        fields = '__all__'


class DatasetViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer

    def perform_create(self, serializer):
        organization, created = Organization.objects.get_or_create(user=self.request.user)
        if created:
            organization.description = "created automatically "
        serializer.save(organization=organization)

    def perform_update(self, serializer):
        """
        Immutable Not allowed
        :param serializer:
        :return:
        """
        raise NotImplementedError

    def perform_destroy(self, instance):
        if instance.organization.user == self.request.user:
            instance.delete()
        else:
            raise ValueError, "User not allowed to delete"


class VDNDetectorViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = VDNRemoteDetector.objects.all()
    serializer_class = VDNRemoteDetectorSerializer

    def perform_create(self, serializer):
        organization, created = Organization.objects.get_or_create(user=self.request.user)
        if created:
            organization.description = "created automatically "
        serializer.save(organization=organization)

    def perform_update(self, serializer):
        """
        Immutable Not allowed
        :param serializer:
        :return:
        """
        raise NotImplementedError

    def perform_destroy(self, instance):
        if instance.organization.user == self.request.user:
            instance.delete()
        else:
            raise ValueError, "User not allowed to delete"


class AnnotationViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = Annotation.objects.all()
    serializer_class = AnnotationSerializer

    def perform_create(self, serializer):
        dataset_id = self.request.POST.get('dataset_id')
        dataset = Dataset.objects.get(pk=dataset_id)
        if dataset.organization.user == self.request.user:
            serializer.save(dataset=dataset)

    def perform_update(self, serializer):
        """
        Immutable Not allowed
        :param serializer:
        :return:
        """
        raise NotImplementedError

    def perform_destroy(self, instance):
        if instance.dataset.organization.user == self.request.user:
            instance.delete()
        else:
            raise ValueError,"User not allowed to delete"


class OrganizationViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = Organization.objects.all()
    serializer_class = OrganizationSerializer


router = routers.DefaultRouter()
router.register(r'datasets', DatasetViewSet)
router.register(r'vdn_detectors', VDNDetectorViewSet)
router.register(r'organizations', OrganizationViewSet)
router.register(r'annotations', AnnotationViewSet)


@register.filter
def get_item(dictionary, key):
    return dictionary[key]


class LoginRequiredMixin(object):
    @method_decorator(login_required)
    def dispatch(self, *args, **kwargs):
        return super(LoginRequiredMixin, self).dispatch(*args, **kwargs)


def marketing(request):
    context = {}
    context['dataset_list'] = Dataset.objects.all()
    context['detector_list'] = VDNRemoteDetector.objects.all()
    context['annotations'] = Annotation.objects.all()
    return render(request, 'vdnapp/vdn_index.html', context=context)


@login_required
def get_token(request):
    context = {}
    context['username'] = request.user.username
    context['token'], created = Token.objects.get_or_create(user=request.user)
    return render(request, 'vdnapp/vdn_token.html', context=context)

