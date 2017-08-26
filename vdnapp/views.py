from django.shortcuts import render
from django.contrib.auth.decorators import login_required,user_passes_test
from django.utils.decorators import method_decorator
from .models import VDNRemoteDataset,Organization,VDNRemoteDetector
from django.template.defaulttags import register
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from rest_framework import serializers, viewsets
from rest_framework import routers
from rest_framework.authtoken.models import Token


class DatasetSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        depth=1
        model = VDNRemoteDataset
        fields = '__all__'


class VDNRemoteDetectorSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        depth=1
        model = VDNRemoteDetector
        fields = '__all__'

class OrganizationSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Organization
        fields = '__all__'


class VDNRemoteDatasetViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = VDNRemoteDataset.objects.all()
    serializer_class = DatasetSerializer


class VDNDetectorViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = VDNRemoteDetector.objects.all()
    serializer_class = VDNRemoteDetectorSerializer


class OrganizationViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = Organization.objects.all()
    serializer_class = OrganizationSerializer


router = routers.DefaultRouter()
router.register(r'vdn_datasets', VDNRemoteDatasetViewSet)
router.register(r'vdn_detectors', VDNDetectorViewSet)
router.register(r'organizations', OrganizationViewSet)


@register.filter
def get_item(dictionary, key):
    return dictionary[key]


class LoginRequiredMixin(object):
    @method_decorator(login_required)
    def dispatch(self, *args, **kwargs):
        return super(LoginRequiredMixin, self).dispatch(*args, **kwargs)


def marketing(request):
    context = {}
    context['dataset_list'] = VDNRemoteDataset.objects.all()
    context['detector_list'] = VDNRemoteDetector.objects.all()
    return render(request, 'vdnapp/vdn_index.html', context=context)



