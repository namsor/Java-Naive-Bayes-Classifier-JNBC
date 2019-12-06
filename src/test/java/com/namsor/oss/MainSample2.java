/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss;

import com.namsor.oss.classify.bayes.ClassifyException;
import com.namsor.oss.classify.bayes.IClassification;
import com.namsor.oss.classify.bayes.NaiveBayesClassifierTransientImpl;
import com.namsor.oss.classify.bayes.PersistentClassifierException;
import java.io.IOException;
import java.io.StringWriter;
import java.util.Arrays;
import java.util.HashSet;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Simple test inspired by
 * https://towardsdatascience.com/introduction-to-na%C3%AFve-bayes-classifier-fa59e3e24aaf
 *
 * @author elian
 */
public class MainSample2 {

    private static final String ZERO = "0";
    private static final String ONE = "1";
    private static final String[] X1 = {"A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C"};
    private static final String[] X2 = {"S", "M", "M", "S", "S", "S", "M", "M", "L", "L", "L", "M", "M", "L", "L"};
    private static final String[] Y = {"0", "0", "1", "1", "0", "0", "0", "1", "1", "1", "1", "1", "1", "1", "0"};

    public static final void main(String[] args) {
        try {
            String[] cats = {ZERO, ONE};
            // Create a new bayes classifier with string categories and string features.
            // INaiveBayesClassifier bayes1 = new NaiveBayesClassifierLevelDBImpl("sentiment", cats, ".", 100);
            NaiveBayesClassifierTransientImpl bayes = new NaiveBayesClassifierTransientImpl("sentiment", cats);
            //NaiveBayesClassifierRocksDBImpl bayes = new NaiveBayesClassifierRocksDBImpl("intro", cats, ".", 100);

// Examples to learn from.
            for (int i = 0; i < Y.length; i++) {
                String x1 = "X1=" + X1[i];
                String x2 = "X2=" + X2[i];
                String[] x12 = {x1, x2};
                bayes.learn(Y[i], new HashSet(Arrays.asList(x12)));
            }

// Here are is X(B,S) to classify.
            String x1 = "X1=B";
            String x2 = "X2=S";
            String[] x12 = {x1, x2};
            IClassification[] predict = bayes.classify(new HashSet(Arrays.asList(x12)));
            StringWriter sw = new StringWriter();
            bayes.dumpDb(sw);
            System.out.println(sw);
            for (int i = 0; i < predict.length; i++) {
                System.out.println("P(" + predict[i].getCategory() + ")=" + predict[i].getProbability());
            }
        } catch (PersistentClassifierException ex) {
            Logger.getLogger(MainSample2.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ClassifyException ex) {
            Logger.getLogger(MainSample2.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(MainSample2.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
