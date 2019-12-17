package com.namsor.oss.classify.bayes;

import java.io.StringWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Simple test inspired by
 * https://www.machinelearningplus.com/predictive-modeling/how-naive-bayes-algorithm-works-with-example-and-full-code/
 *
 * @author elian
 */
public class MainSample3 {

    public static final String BANANA = "Banana";
    public static final String ORANGE = "Orange";
    public static final String OTHER = "Other";

    /**
     * Header table as per
     * https://www.machinelearningplus.com/predictive-modeling/how-naive-bayes-algorithm-works-with-example-and-full-code/
     */
    public static final String[] colName = {
        "Fruit", "Long", "Sweet", "Yellow", "Weight"
    };

    /**
     * Data table as per
     * https://www.machinelearningplus.com/predictive-modeling/how-naive-bayes-algorithm-works-with-example-and-full-code/
     */
    public static final String[][] data = {
        {"Banana", "Yes", "Yes", "Yes", "350"},
        {"Banana", "Yes", "No", "Yes", "50"},
        {"Banana", "No", "No", "Yes", "50"},
        {"Banana", "No", "No", "No", "50"},
        {"Orange", "No", "Yes", "Yes", "150"},
        {"Orange", "No", "No", "Yes", "150"},
        {"Other", "Yes", "Yes", "Yes", "50"},
        {"Other", "Yes", "Yes", "No", "50"},
        {"Other", "No", "Yes", "No", "50"},
        {"Other", "No", "No", "No", "50"},
    };

    public static final void main(String[] args) {

        try {
            String[] cats = {BANANA, ORANGE, OTHER};
            // Create a new bayes classifier with string categories and string features.
            NaiveBayesClassifierTransientImpl bayes = new NaiveBayesClassifierTransientImpl("fruit", cats);
            //NaiveBayesClassifierTransientLaplacedImpl bayes = new NaiveBayesClassifierTransientLaplacedImpl("fruit", cats);
            //NaiveBayesClassifierRocksDBImpl bayes = new NaiveBayesClassifierRocksDBImpl("intro", cats, ".", 100);

            // Examples to learn from.
            for (int i = 0; i < data.length; i++) {
                Map<String, String> features = new HashMap();
                features.put(colName[1], data[i][1]);
                features.put(colName[2], data[i][2]);
                features.put(colName[3], data[i][3]);
                bayes.learn(data[i][0], features, Long.parseLong(data[i][4]));
            }
            /**
             * Calculate the likelihood that: Long, Sweet, Yellow is a Banana
             */

            // Here are is X(B,S) to classify.
            Map<String, String> features = new HashMap();
            features.put("Long", "Yes");
            features.put("Sweet", "Yes");
            features.put("Yellow", "Yes");
            IClassification[] predict = bayes.classify(features);
            StringWriter sw = new StringWriter();
            bayes.dumpDb(sw);
            System.out.println(sw);
            for (int i = 0; i < predict.length; i++) {
                System.out.println("P(" + predict[i].getCategory() + ")=" + predict[i].getProbability());
            }
        } catch (PersistentClassifierException ex) {
            Logger.getLogger(MainSample1Laplaced.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ClassifyException ex) {
            Logger.getLogger(MainSample1Laplaced.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Throwable ex) {
            Logger.getLogger(MainSample1Laplaced.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
