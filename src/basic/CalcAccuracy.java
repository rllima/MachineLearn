package basic;

import java.util.List;

public class CalcAccuracy {
    public double calcAccuracy(String[] predictions, List<Instance> test){
        double value;
        double pos = 0;
        double neg = 0;
        for (int i = 0; i <test.size(); i++)
        {
            if (test.get(i).getPrediction().equalsIgnoreCase(predictions[i]))
                pos++;
            else
                neg++;
        }
        value =  (pos / (pos+neg));
        return value;
    }
}
