import pickle
import matplotlib.pyplot as plt

def predict(query):
    loaded_model = pickle.load(open("MultinomialNBModel2", 'rb'))
    countVectorizer = pickle.load(open("MultinomialNBCountVectorizer2", "rb"))

    vectorizedQuery = countVectorizer.transform([query]).toarray()


    prediction = loaded_model.predict(vectorizedQuery)
    probabilities = loaded_model.predict_proba([vectorizedQuery[0]])[0]
    probabilities = [float(x) * 100 for x in probabilities]
    return prediction, probabilities



def generateGraphs(numbersList, name, xlabel, ylabel, title):
    fig = plt.figure()

    movingAverages = getMovingAverage(numbersList)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlabel(xlabel)
    # plt.yticks(range(0, 100), range(0, 100))
    # plt.xticks(range(0, len(numbersList)))
    # plt.ylim((0, 100))
    # plt.xlim((0, len(numbersList)))
    plt.axis([0, len(numbersList), 0, 100])
    ax = plt.axes()



    # ax.xaxis.set_ticks(range(len(numbersList)))
    # ax.xaxis.set_ticklabels(range(len(numbersList)))
    # plt.xticks(rotation=90)
    ax.set_xticks([])
    ax.yaxis.set_ticks(range(100, 10))
    ax.yaxis.set_ticks(range(100, 10))

    # ax.figure.autofmt_xdate()
    plt.plot(numbersList)
    plt.plot(movingAverages)

    print(numbersList)
    print(movingAverages)
    plt.savefig(name, format='svg')


def getMovingAverage(numbersList):
    averageList = []
    alpha = 0.9
    movingAverage = numbersList[0]

    averageList.append(movingAverage)

    for number in numbersList[1:]:
        movingAverage = alpha * movingAverage + (1-alpha) * number
        averageList.append(movingAverage)

    return averageList