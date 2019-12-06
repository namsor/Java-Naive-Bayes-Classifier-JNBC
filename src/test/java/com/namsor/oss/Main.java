/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss;

import com.namsor.oss.classify.bayes.ClassifyException;
import com.namsor.oss.classify.bayes.INaiveBayesClassifier;
import com.namsor.oss.classify.bayes.NaiveBayesClassifierLevelDBImpl;
import com.namsor.oss.classify.bayes.NaiveBayesClassifierRocksDBImpl;
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
 * https://github.com/ptnplanet/Java-Naive-Bayes-Classifier
 *
 * @author elian
 */
public class Main {

    private static final String POSITIVE = "positive";
    private static final String NEGATIVE = "negative";

    public static final void main(String[] args) {
        try {
            String[] cats = {POSITIVE, NEGATIVE};
            // Create a new bayes classifier with string categories and string features.
            // INaiveBayesClassifier bayes1 = new NaiveBayesClassifierLevelDBImpl("sentiment", cats, ".", 100);
            //INaiveBayesClassifier bayes2 = new NaiveBayesClassifierTransientImpl("sentiment", cats);
            NaiveBayesClassifierRocksDBImpl bayes = new NaiveBayesClassifierRocksDBImpl("sentiment", cats, ".", 100);
            StringWriter sw = new StringWriter();
            bayes.dumpDb(sw);
            System.out.println("Bayes Stats 1 : \n"+sw);
            
// Two examples to learn from.
            String[] positiveText = "I love sunny days".split("\\s");
            String[] negativeText = "I hate rain".split("\\s");

// Learn by classifying examples.
// New categories can be added on the fly, when they are first used.
// A classification consists of a category and a list of features
// that resulted in the classification in that category.
            bayes.learn(POSITIVE, new HashSet(Arrays.asList(positiveText)));
            bayes.learn(NEGATIVE, new HashSet(Arrays.asList(negativeText)));

// Here are two unknown sentences to classify.
            String[] unknownText1 = "today is a sunny day".split("\\s");
            String[] unknownText2 = "there will be rain".split("\\s");
            sw = new StringWriter();
            bayes.dumpDb(sw);
            System.out.println("Bayes Stats 2 : \n"+sw);
            System.out.println(sw);
            System.out.println( // will output "positive"
                    bayes.classify(new HashSet(Arrays.asList(unknownText1)))[0].getCategory());
            System.out.println( // will output "negative"
                    bayes.classify(new HashSet(Arrays.asList(unknownText2)))[0].getCategory());

        } catch (PersistentClassifierException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ClassifyException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
