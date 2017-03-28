from django import forms
from django.forms import ModelForm
from models import VLabel


class UploadFileForm(forms.Form):
    name = forms.CharField()
    file = forms.FileField()


class YTVideoForm(forms.Form):
    name = forms.CharField()
    url = forms.CharField()


class AnnotationForm(forms.Form):
    x = forms.FloatField()
    y = forms.FloatField()
    h = forms.FloatField()
    w = forms.FloatField()
    detection = forms.IntegerField()
    metadata = forms.CharField(required=False)
    tags = forms.CharField(required=False)
    high_level = forms.BooleanField(required=False)


class VLabelForm(ModelForm):
    class Meta:
        model = VLabel
        fields = ['label_name']

