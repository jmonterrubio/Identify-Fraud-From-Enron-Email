### Characteristics of the dataset

def obtain_features(data):
    features = []
    for key in data:
        for feature in data[key]:
            if not feature in features:
                features.append(feature)
    return features

def count_feature(name, data):
    feature = 0
    for key in data:
        if data[key][name] != 'NaN':
            feature +=1
    return feature

def summary_features(data):
    countFeatures = {}
    features = obtain_features(data)
    for feature in features:
        countFeatures[feature] = count_feature(feature, data)
    return countFeatures

def feature_summary(name, data):
    summary = {}
    for key in data:
        if summary.has_key(data[key][name]):
            summary[data[key][name]] += 1
        else:
            summary[data[key][name]] = 1
    return summary
    
def summary(data_dict):
    print "Total number of datapoints: ", len(data_dict)
    print "Total number of features: ", len(obtain_features(data_dict))
    print "Datapoint names: ", data_dict.keys()
    print "Summary features: ", summary_features(data_dict)
    print "Label class poi: ", feature_summary('poi', data_dict)
