from django import forms
from django.forms import ModelForm


class UploadFileForm(forms.Form):
    name = forms.CharField()
    file = forms.FileField()
    scene = forms.BooleanField(initial=False,required=False)


class YTVideoForm(forms.Form):
    name = forms.CharField()
    url = forms.CharField()
    scene = forms.BooleanField(initial=False, required=False)


class AnnotationForm(forms.Form):
    x = forms.FloatField()
    y = forms.FloatField()
    h = forms.FloatField()
    w = forms.FloatField()
    object_name = forms.CharField()
    text = forms.CharField(required=False)
    metadata = forms.CharField(required=False)
    tags = forms.CharField(required=False)
    high_level = forms.BooleanField(required=False)



