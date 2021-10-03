import os
from flask import Flask, render_template, request, url_for
from flask_bootstrap import Bootstrap
from test import predict, generateGraphs
import time
import re

rawText = ""
OverAnalysisGoodScore = 0.0
OverAnalysisRacismScore = 0.0
OverAnalysisSexismScore = 0.0
FinalClassification="NA"
SentenceAnalysis=[]


OverallOverAnalysisRacismScore = 0
OverallOverAnalysisSexismScore = 0
OverallOverAnalysisGoodScore = 0
OverallFinalClassification = "NA"


app = Flask(__name__)
Bootstrap(app)
@app.route('/')
def index():
    print("hello")
    return render_template("./index.html",rawText=rawText, OverAnalysisRacismScore=OverallOverAnalysisRacismScore, OverAnalysisSexismScore=OverallOverAnalysisSexismScore,SentenceAnalysis=SentenceAnalysis, OverAnalysisGoodScore=OverallOverAnalysisGoodScore,FinalClassification=OverallFinalClassification)

@app.route('/analyze', methods=['POST'])
def analyze():

    # to do
    if request.method == 'POST':
        rawText = request.form['rawText']

    sentences = []
    OverAnalysisRacismScores = []
    OverAnalysisSexismScores = []
    OverAnalysisGoodScores = []
    FinalClassificationsList = []


    for sentence in re.split('[.?!]', rawText):
        if len(sentence.strip()) == 0:
            continue

        print("SENTENCE IS")
        print(sentence)
        # e.g. classification = 'sexist', probabilies = [93, 6, 1]
        classification, probabilities = predict(sentence)

        if classification == "non-rascist/sexist":
            classification = ["non-racist/sexist"]

        OverAnalysisRacismScore = probabilities[1]
        # OverAnalysisRacismConfidence =
        OverAnalysisSexismScore = probabilities[2]
        # OverAnalysisSexismConfidence = -1
        OverAnalysisGoodScore = probabilities[0]

        FinalClassification = classification[0]

        if len(sentence) > 28:
            tempSentence = sentence[:28] + "..."
        else:
            tempSentence = sentence

        sentences.append(tempSentence)
        OverAnalysisRacismScores.append(OverAnalysisRacismScore)
        OverAnalysisSexismScores.append(OverAnalysisSexismScore)
        OverAnalysisGoodScores.append(OverAnalysisGoodScore)
        FinalClassificationsList.append(FinalClassification)

    # e.g. classification = 'sexist', probabilies = [93, 6, 1]
    classification, probabilities = predict(rawText)

    if classification == "non-rascist/sexist":
        classification = ["non-racist/sexist"]

    OverallOverAnalysisRacismScore = probabilities[1]
    # OverAnalysisRacismConfidence =
    OverallOverAnalysisSexismScore = probabilities[2]
    # OverAnalysisSexismConfidence = -1
    OverallOverAnalysisGoodScore = probabilities[0]

    OverallFinalClassification = classification[0]


    print("OVERALL")
    print(OverallOverAnalysisSexismScore)
    print(OverallOverAnalysisRacismScore)
    print(OverallOverAnalysisGoodScore)
    print(OverallFinalClassification)
    print(rawText)

    print()
    print("INDIVIDUALS")
    print(sentences)
    print(OverAnalysisRacismScores)
    print(OverAnalysisSexismScores)
    print(OverAnalysisGoodScores)
    print(FinalClassificationsList)


    generateGraphs(OverAnalysisRacismScores, 'static/images/racism.svg', 'sentence', 'racism score', "Racism Scores Timeline")
    generateGraphs(OverAnalysisSexismScores, 'static/images/sexism.svg', 'sentence', 'sexism score', "Sexism Scores Timeline")
    generateGraphs(OverAnalysisGoodScores, 'static/images/good.svg', 'sentence', 'good score', "Non-Racism/Sexism Scores Timeline")

    time.sleep(1)

    SentenceAnalysis = [{"id": index+1, "Sentence": sentence, "Racism": racismScore, "Sexism": sexismScore, "Good": goodScore, "Final Classification": classification} for
                        index, (sentence, racismScore, sexismScore, goodScore, classification) in enumerate(zip(sentences, OverAnalysisRacismScores, OverAnalysisSexismScores, OverAnalysisGoodScores, FinalClassificationsList))]

    # SentenceAnalysis=[
    # {"id": 1, "Sentence": sentences, "Racism": OverAnalysisRacismScores, "Sexism": OverAnalysisSexismScores, "Good": OverAnalysisGoodScores, "Final Classification": FinalClassificationsList}]
    return render_template("./index.html",rawText=rawText, OverAnalysisRacismScore=OverallOverAnalysisRacismScore,OverAnalysisSexismScore=OverallOverAnalysisSexismScore,SentenceAnalysis=SentenceAnalysis, OverAnalysisGoodScore=OverallOverAnalysisGoodScore,FinalClassification=OverallFinalClassification)

if __name__=="__main__":
    app.run(debug=True)

