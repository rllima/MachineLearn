package KNNImplementation;

import basic.CalcEuclidianDistance;
import basic.Distance;
import basic.Instance;

import java.util.*;

public class wKNN {
    private int k;
    private int totalTrainSet;
    private List<Instance> trainSet;
    private List<Instance> customerSet;

    public wKNN(int k, int totalTrainSet, List<Instance> trainSet, List<Instance> customerSet) {
        this.k = k;
        this.totalTrainSet = totalTrainSet;
        this.trainSet = trainSet;
        this.customerSet = customerSet;

    }
    public long runWKNN(int distance)
    {
        long start;
        long delay;
        start = System.currentTimeMillis();
        Distance[] distances = new Distance[customerSet.size()];
        for (int i = 0; i < this.customerSet.size(); i++)
        {
           switch (distance){
               case 1:
                   distances = new CalcEuclidianDistance().euclidianDistance(this.customerSet.get(i),this.trainSet);
                   break;
           }
           setPredictionPondered(calDistancePondered(distances),this.customerSet.get(i));
        }
        delay = System.currentTimeMillis() - start;

        return delay;

    }

    public Distance[] calDistancePondered(Distance[] distance){
        Distance[] aux = new Distance[k];
        for(int i = 0; i < k; i++){
            double x = distance[i].getDistance();
            aux[i] = new Distance();
            aux[i].index = distance[i].getIndex();
            aux[i].distance = ((1/Math.max(Math.pow(x,2), 0.0000000000000001)));
        }
        return aux;
    }

    public void setPredictionPondered(Distance[] distance, Instance instance){
        HashMap<String,Double> fr = new HashMap<>();
        for(int j = 0; j < k; j++) {
            int index = distance[j].getIndex();
            String prediction = this.trainSet.get(index).getPrediction();
            if(fr.containsKey(prediction)) {
                double accum = fr.get(prediction);
                accum += distance[j].distance;
                fr.replace(prediction,accum);
            }else{
                fr.put(prediction,0.0);
            }
        }

        double max = -1;
        String res="";
        for(String key : fr.keySet()){
            if(fr.get(key)> max ){
                max = fr.get(key);
                res = key;
            }
        }
        instance.setPrediction(res);
    }

}
