import pandas as pd
from scipy.io import arff


def readBase():
    data = arff.loadarff('F:\AM\Listas\MachineLearn\List02\DataBases\KC1.arff')
    df = pd.DataFrame(data[0])
    dataset = []
    for i,instance in enumerate(df.values):
        dataset.append([])
        for j,attribute in enumerate(instance):
            dataset[i].append(attribute)

    return dataset
    

            
    
        