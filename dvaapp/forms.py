from django import forms
from django.forms import ModelForm


class UploadFileForm(forms.Form):
    name = forms.CharField()
    file = forms.FileField()
    scene = forms.BooleanField(initial=False,required=False)
    nth = forms.IntegerField()
    rescale = forms.IntegerField(required=False)

class YTVideoForm(forms.Form):
    name = forms.CharField()
    url = forms.CharField()


class AnnotationForm(forms.Form):
    x = forms.FloatField()
    y = forms.FloatField()
    h = forms.FloatField()
    w = forms.FloatField()
    metadata_text = forms.CharField(required=False)
    metadata_json = forms.CharField(required=False)
    tags = forms.CharField(required=False)
    high_level = forms.BooleanField(required=False)



