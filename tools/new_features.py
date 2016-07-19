def computeFraction( poi_messages, all_messages ):
    fraction = 0.
    if float(poi_messages) > 0 and float(all_messages) > 0:
        fraction = float(poi_messages) / float(all_messages)

    return fraction

def addFractionFeature(data_dict, feature_name, origin1, origin2):
    submit_dict = {}
    for name in data_dict:
        data_point = data_dict[name]
        data_point[feature_name] = computeFraction(data_point[origin1], data_point[origin2])