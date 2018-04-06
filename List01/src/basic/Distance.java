package basic;

public class Distance implements Comparable<Distance> {
    public int index;
    public double distance;

    public int getIndex() {
        return index;
    }
    public void setIndex(int index){
        this.index = index;
    }
    public double getDistance() {
        return distance;
    }
    public void setDistance(double distance) {
        this.distance = distance;
    }

    @Override
    public int compareTo(Distance o) {
      return Double.compare(this.distance, o.distance);
    }
}
