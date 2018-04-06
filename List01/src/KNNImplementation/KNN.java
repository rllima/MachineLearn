package KNNImplementation;
import basic.CalcEuclidianDistance;
import basic.Distance;
import basic.Instance;

import java.util.*;

public class KNN {

    private int k;
    private int totalTrainSet;
    private List<Instance> trainSet;
    private List<Instance> customerSet;

    public KNN(int k, int totalTrainSet, List<Instance> trainSet, List<Instance> customerSet) {
        this.k = k;
        this.totalTrainSet = totalTrainSet;
        this.trainSet = trainSet;
        this.customerSet = customerSet;

    }
    public long runkNN(int distance)
    {
        long start;
        long delay;
        start = System.currentTimeMillis();
        Distance[] distances = new Distance[this.customerSet.size()];
        for (int i = 0; i < this.customerSet.size(); i++)
        {
            switch (distance) {
                case (1):
                    distances = new CalcEuclidianDistance().euclidianDistance(this.customerSet.get(i),this.trainSet);
                    break;
                default:
            }
            setPrediction(frequencyVote(distances),this.customerSet.get(i));
        }
        delay = System.currentTimeMillis() - start;

        return delay;
    }

    public void setPrediction(String pred,Instance instance){
        instance.setPrediction(pred);
    }

    public String frequencyVote(Distance[] distance){
        HashMap<String,Integer> frequenccy = new HashMap<>();
        for(int i = 0; i < k; i++){
            Instance tmp = this.trainSet.get(distance[i].getIndex());
            if(frequenccy.containsKey(tmp.getPrediction())){
                int fr = frequenccy.get(tmp.getPrediction());
                frequenccy.put(tmp.getPrediction(),++fr);
            }else
                frequenccy.put(tmp.getPrediction(), 1);
        }

        int max = -1;
        String pred = "";
        for(String resp : frequenccy.keySet()){
            if(frequenccy.get(resp)> max){
                max = frequenccy.get(resp);
                pred = resp;
            }
        }
        return pred;
    }

}
