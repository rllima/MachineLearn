

import basic.CalcAccuracy;
import KNNImplementation.KNN;
import KNNImplementation.wKNN;
import basic.Instance;

import basic.Pair;
import weka.core.Instances;

import java.io.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class KnnProgramm {

    public static void main(String[] args) throws IOException {

        int seed = 1;          // the seed for randomizing the data
        int folds = 10;
        int k = 9;             // Variação do K
        Random rand = new Random(seed);   // create seeded number generator
        Instances data = new Instances(new FileReader("src/Databases/KC1.txt")); // Alterar para um dos nome contidos no package Databases
        String name = data.relationName();
        Instances randData = new Instances(data);
        randData.randomize(rand);

        double accuracyKNN = 0;
        double accuracyWKNN = 0;
        long timeKNN = 0;
        long timewKK = 0;

        for (int n = 1; n < folds; n++) {
            Instances train = randData.trainCV(folds, n);
            Instances test = randData.testCV(folds, n);

        //+++++++++++++++   NORMALIZAÇÂO - MAXIMO E MINIMO   +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            int randomNum = ThreadLocalRandom.current().nextInt(0, train.numInstances());
            boolean isNumeric = false;
            for (int i = 0; i < train.numAttributes() - 1 ; i++) {

                if(train.instance(randomNum).attribute(i).isNumeric()){
                    isNumeric = true;
                }else
                    isNumeric = false;
            }
            if(isNumeric){
                double [] max = train.instance(0).toDoubleArray();
                double [] min = train.instance(1).toDoubleArray();
                for (int i = 0; i < train.numAttributes() - 1 ; i++) {
                    for (int j = 0; j < train.numInstances() ; j++) {
                        if(train.instance(j).value(i) > max[i])
                            max[i] = train.instance(j).value(i);
                        if(train.instance(j).value(i) < min[i])
                            min[i] = train.instance(j).value(i);
                    }
                }
                for (int i = 0; i < train.numAttributes() - 1 ; i++) {
                    for (int j = 0; j < train.numInstances() ; j++) {
                       double aux = train.instance(j).value(i);
                       train.instance(j).setValue(i,((aux - min[i])/ (max[i]-min[i])));

                    }
                }


            }
        //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


            List<Instance> trainSet = new ArrayList<>();
            List<Instance> testSet = new ArrayList<>();

            String[] responses = new String[test.numInstances()];

            for (int i = 0; i < train.numInstances(); i++) {
                Pair<Double,String> [] pair = new Pair[train.numAttributes()];
                for (int j = 0; j < train.numAttributes() ; j++) {
                    if(train.instance(i).attribute(j).isNumeric()){
                        double value = train.instance(i).value(j);
                        pair[j] = new Pair<Double,String>(value,"");
                    }else{
                        pair[j] = new Pair<Double,String>(1.0, train.instance(i).stringValue(j));
                    }
                }
                trainSet.add(new Instance(pair,train.instance(i).stringValue(train.numAttributes()-1)));
            }
            for (int i = 0; i < test.numInstances(); i++) {
                Pair [] pair = new Pair[test.numAttributes()];
                for (int j = 0; j < test.numAttributes() ; j++) {
                    if(test.instance(i).attribute(j).isNumeric()){
                        double value = train.instance(i).value(j);
                        pair[j] = new Pair<Double,String>(value,"");
                    }else{
                        pair[j] = new Pair<Double,String>(1.0, test.instance(i).stringValue(j));
                    }
                }
                testSet.add(new Instance(pair,train.instance(i).stringValue(test.numAttributes()-1)));
                responses[i] = test.instance(i).stringValue(test.numAttributes()-1);
            }


            KNN knn = new KNN(k, trainSet.size(), trainSet, testSet);
            timeKNN += knn.runkNN(1);
            accuracyKNN += new CalcAccuracy().calcAccuracy(responses,testSet);

            wKNN wKnn = new wKNN(k, trainSet.size(), trainSet, testSet);
            timewKK += wKnn.runWKNN(1);
            accuracyWKNN += new CalcAccuracy().calcAccuracy(responses,testSet);

        }
        System.out.println("Final Results:\n" +
                "___________________________________________________________________________________\n"+
                "Database: | " + name+" |\n"+
                "KNN:      | " + k + "  |K-Fold Cross-Validation: " + folds+"|\n"+
                "Accuracy  | " + accuracyKNN/(double)folds + "|Time: "+timeKNN+" milliseconds|\n"+
                "===================================================================================\n\n");

        System.out.println("Final Results:\n" +
                "___________________________________________________________________________________\n"+
                "Database: | " + name+"|\n"+
                "WKNN:     | " + k + "   |K-Fold Cross-Validation: " + folds+" |\n"+
                "Accuracy  | " + accuracyWKNN/(double)folds + "|Time: "+timewKK+" milliseconds |\n"+
                "===================================================================================\n\n");


    }



}


