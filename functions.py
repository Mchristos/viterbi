def normalize(array):
    normal = []
    for element in array:
        normal.append(element/sum(array))
    return normal
