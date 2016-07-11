### Find outliers help
import matplotlib.pyplot

def scatterplot(data):
    for point in data:
        feature1 = point[0]
        feature2 = point[1]
        matplotlib.pyplot.scatter( feature1, feature2 )

    matplotlib.pyplot.xlabel("feature1")
    matplotlib.pyplot.ylabel("feature2")
    matplotlib.pyplot.show()

def findTopOutliersPair(data, N=1):
    sorted_data = sorted(data, key = lambda data:data[0])
    return sorted_data[len(data)-N:len(data)]

def removeTopOutlier(data): 
    sorted_data = sorted(data, key = lambda data:data[0])
    return sorted_data[0:len(data)-1]

def findTopOutliersKey(data_dict, feature, value):
    return [k for k in data_dict.keys() if data_dict[k][feature] == value]