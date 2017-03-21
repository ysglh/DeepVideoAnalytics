from django import forms


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
    metadata = forms.CharField(required=False)
    name = forms.CharField()

