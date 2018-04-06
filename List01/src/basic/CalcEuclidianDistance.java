package basic;

import java.util.Arrays;
import java.util.List;

public class CalcEuclidianDistance {
    public Distance[] euclidianDistance(Instance instance, List<Instance> totalTrainSet){
        Distance [] distance = new Distance[totalTrainSet.size()];
        for (int i = 0; i < totalTrainSet.size(); i++)
        {
            distance[i] = new Distance();
            distance[i].setDistance(0.0);
            distance[i].setIndex(i);
            distance[i].setDistance(calcEuclides(instance.getAttributes(), totalTrainSet.get(i).getAttributes()));
        }
        Arrays.sort(distance);
        return distance;
    }

    public Double calcEuclides(Pair[] a, Pair[] b)
    {
        double distance = 0;
        for (int i = 0; i < a.length-1; i++)
        {
            distance += Math.pow(( (double)a[i].first - (double)b[i].first),2);
        }
        return Math.sqrt(distance);
    }

}
