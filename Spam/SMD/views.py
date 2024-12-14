
from django.shortcuts import render
from django.http import HttpResponse
import os
import joblib

# Load models
model1 = joblib.load(os.path.dirname(__file__) + "//RandomForest.pkl")
model2 = joblib.load(os.path.dirname(__file__) + "//SpportVectorMachine.pkl")
#model3 = joblib.load(os.path.dirname(__file__)+ "//DecisionTreeClassifier.pkl")
model3 = joblib.load(os.path.dirname(__file__)+ "//NaiveBayes.pkl")
# Create your views here.
def index(request):
    return render(request, 'index.html')

def checkSpam(request):
    if request.method == "POST":
        finalAns = ""
        algo = request.POST.get("algo")
        rawData = request.POST.get("rawdata")

        if algo == "Algo-1":
            finalAns = model1.predict([rawData])[0]  # Correct variable name 'rawData'
        elif algo == "Algo-2":
            finalAns = model2.predict([rawData])[0]  # Correct variable name 'rawData'
        elif algo == "Algo-3":
            finalAns = model3.predict([rawData])[0]
        param = {"answer": finalAns}
        return render(request, 'output.html', param)
    else:
        return render(request, 'index.html')    