/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

import static com.namsor.oss.classify.bayes.MainSample1.NO;
import static com.namsor.oss.classify.bayes.MainSample1.YES;
import static com.namsor.oss.classify.bayes.MainSample1.colName;
import static com.namsor.oss.classify.bayes.MainSample1.data;
import static com.namsor.oss.classify.bayes.MainSample2.ONE;
import static com.namsor.oss.classify.bayes.MainSample2.X1;
import static com.namsor.oss.classify.bayes.MainSample2.X2;
import static com.namsor.oss.classify.bayes.MainSample2.Y;
import static com.namsor.oss.classify.bayes.MainSample2.ZERO;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.Map;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author elian
 */
public class NaiveBayesClassifierTransientImplV2Test {

    public NaiveBayesClassifierTransientImplV2Test() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

    /**
     * Test based on https://taylanbil.github.io/boostedNB or
     * http://ai.fon.bg.ac.rs/wp-content/uploads/2015/04/ML-Classification-NaiveBayes-2014.pdf
     */
    @Test
    public void testTransientLearnClassifySample1() throws Exception {
        String[] cats = {YES, NO};
        NaiveBayesClassifierTransientImpl bayes = new NaiveBayesClassifierTransientImpl("tennis", cats);
        for (int i = 0; i < data.length; i++) {
            Map<String, String> features = new HashMap();
            for (int j = 0; j < colName.length - 1; j++) {
                features.put(colName[j], data[i][j]);
            }
            bayes.learn(data[i][colName.length - 1], features);
        }
        Map<String, String> features = new HashMap();
        features.put("outlook", "Sunny");
        features.put("temp", "Cool");
        features.put("humidity", "High");
        features.put("wind", "Strong");
        IClassification[] predict = bayes.classify(features);
        assertNotNull(predict);
        assertEquals(predict.length, 2);
        assertEquals(predict[0].getCategory(), "No");
        assertEquals(predict[1].getCategory(), "Yes");
        assertEquals(predict[0].getProbability(), 0.795417348608838, .0001);
        assertEquals(predict[1].getProbability(), 0.204582651391162, .0001);
    }

    /**
     * Test based on
     * https://towardsdatascience.com/introduction-to-na%C3%AFve-bayes-classifier-fa59e3e24aaf
     */
    @Test
    public void testTransientLearnClassifySample2() throws Exception {
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
        assertNotNull(predict);
        assertEquals(predict.length, 2);
        assertEquals(predict[0].getCategory(), "0");
        assertEquals(predict[1].getCategory(), "1");
        assertEquals(predict[0].getProbability(), 0.75, .0001);
        assertEquals(predict[1].getProbability(), 0.25, .0001);
    }

}
