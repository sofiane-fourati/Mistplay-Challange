from django import forms

class FilesForm(forms.Form):
	file = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': False}))