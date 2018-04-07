import pandas as pd
from scipy.io import arff


def readBase():
    data = arff.loadarff('F:\AM\Listas\MachineLearn\List02\DataBases\KC1.arff')
    df = pd.DataFrame(data[0])
    return df
    

            
    
        