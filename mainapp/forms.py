from django import forms


class UploadForm(forms.Form):  # 파일 or url 입력
    file = forms.FileField(required=False)
    url = forms.URLField(required=False)


class EvaluateForm(forms.Form):  # 평가 입력
    CHOICES = [('G', 'Good'), ('B', 'Bad')]  # ('value', 'shown on the radio button')
    option = forms.ChoiceField( widget=forms.RadioSelect, choices=CHOICES)
    eval_text = forms.CharField(required=False, widget=forms.Textarea())

