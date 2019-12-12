package com.namsor.oss.classify.bayes;

import com.namsor.oss.classify.bayes.ClassifyException;
import com.namsor.oss.classify.bayes.IClassification;
import com.namsor.oss.classify.bayes.NaiveBayesClassifierTransientImpl;
import java.io.IOException;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Simple test inspired by
 * https://towardsdatascience.com/introduction-to-na%C3%AFve-bayes-classifier-fa59e3e24aaf
 *
 * @author elian
 */
public class MainSample2 {

    public static final String ZERO = "0";
    public static final String ONE = "1";
    public static final String[] X1 = {"A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C"};
    public static final String[] X2 = {"S", "M", "M", "S", "S", "S", "M", "M", "L", "L", "L", "M", "M", "L", "L"};
    public static final String[] Y = {"0", "0", "1", "1", "0", "0", "0", "1", "1", "1", "1", "1", "1", "1", "0"};

    public static final void main(String[] args) {
        try {
            String[] cats = {ZERO, ONE};
            // Create a new bayes classifier with string categories and string features.
            // INaiveBayesClassifier bayes1 = new NaiveBayesClassifierLevelDBImpl("sentiment", cats, ".", 100);
            NaiveBayesClassifierTransientImpl bayes = new NaiveBayesClassifierTransientImpl("sentiment", cats);
            //NaiveBayesClassifierRocksDBImpl bayes = new NaiveBayesClassifierRocksDBImpl("intro", cats, ".", 100);

// Examples to learn from.
            for (int i = 0; i < Y.length; i++) {
                Map<String, String> features = new HashMap();
                features.put("X1", X1[i]);
                features.put("X2", X2[i]);
                bayes.learn(Y[i], features);
            }

// Here are is X(B,S) to classify.
            Map<String, String> features = new HashMap();
            features.put("X1", "B");
            features.put("X2", "S");
            IClassification[] predict = bayes.classify(features);
            StringWriter sw = new StringWriter();
            bayes.dumpDb(sw);
            System.out.println(sw);
            for (int i = 0; i < predict.length; i++) {
                System.out.println("P(" + predict[i].getCategory() + ")=" + predict[i].getProbability());
            }
        } catch (ClassifyException ex) {
            Logger.getLogger(MainSample2.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Throwable ex) {
            Logger.getLogger(MainSample2.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
