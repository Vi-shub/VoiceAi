from django.shortcuts import render
from .models import MeetFile

# Create your views here.

def home(request):
    return render(request, 'index.html', {})

def uploadpage(request):
    if request.method == 'POST':
        file = request.FILES['file']
        mf = MeetFile(file=file)
        mf.save()
    return render(request, 'uploadpage.html', {})


