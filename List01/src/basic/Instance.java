package basic;



public class Instance {
    Pair <Double,String>[] attributes;
    String prediction;

    public Instance(Pair<Double,String>[] attributes, String prediction) {
        this.attributes = attributes;
        this.prediction = prediction;
    }


    public Pair<Double,String>[] getAttributes() {
        return attributes;
    }

    public String getPrediction(){
        return this.prediction;
    }

    public void setPrediction(String pred){
        this.prediction = pred;
    }
}
