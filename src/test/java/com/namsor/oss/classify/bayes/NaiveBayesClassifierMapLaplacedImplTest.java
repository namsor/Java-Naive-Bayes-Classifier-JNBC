package com.namsor.oss.classify.bayes;



import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static com.namsor.oss.samples.MainSample1.*;
import static com.namsor.oss.samples.MainSample2.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import org.junit.jupiter.api.BeforeEach;

/**
 * todo remove comment
 *
 * @author elian
 */
public class NaiveBayesClassifierMapLaplacedImplTest {
    private static final String MAPDB_DIR = "/tmp/mapdb";

    public NaiveBayesClassifierMapLaplacedImplTest() {
    }

    @BeforeEach
    public void setUp() {
        File mapdb = new File(MAPDB_DIR);
        if (mapdb.exists() && mapdb.isDirectory()) { 
            // ok
        } else {
            mapdb.mkdirs();
        }
    }


    /**
     * Test based on https://taylanbil.github.io/boostedNB or
     * http://ai.fon.bg.ac.rs/wp-content/uploads/2015/04/ML-Classification-NaiveBayes-2014.pdf
     */
    @Test
    public void testLearnClassifySample1() throws Exception {
        String[] cats = {YES, NO};
        NaiveBayesClassifierMapLaplacedImpl bayes = new NaiveBayesClassifierMapLaplacedImpl("tennis", cats, 1, false);
        for (int i = 0; i < data.length; i++) {
            Map<String, String> features = new HashMap();
            for (int j = 0; j < colName.length - 1; j++) {
                features.put(colName[j], data[i][j]);
            }
            bayes.learn(data[i][colName.length - 1], features);
        }
        Map<String, String> features = new HashMap();
        features.put("outlook", "Overcast");
        features.put("temp", "Cool");
        features.put("humidity", "High");
        features.put("wind", "Strong");
        IClassification predict = bayes.classify(features, true);
        assertNotNull(predict);
        assertEquals(predict.getClassProbabilities().length, 2);
        assertEquals(predict.getClassProbabilities()[0].getCategory(), "Yes");
        assertEquals(predict.getClassProbabilities()[1].getCategory(), "No");
        assertEquals(predict.getClassProbabilities()[0].getProbability(), 0.7215830648872527, .0001);
        assertEquals(predict.getClassProbabilities()[1].getProbability(), 0.2784169351127473, .0001);
    }


    /**
     * Test based on https://taylanbil.github.io/boostedNB or
     * http://ai.fon.bg.ac.rs/wp-content/uploads/2015/04/ML-Classification-NaiveBayes-2014.pdf
     */
    @Test
    public void testLearnClassifyMapDBSample1() throws Exception {
        String[] cats = {YES, NO};
        NaiveBayesClassifierMapLaplacedImpl bayes = new NaiveBayesClassifierMapLaplacedImpl("tennis", cats, 1, false, MAPDB_DIR + System.currentTimeMillis());
        for (int i = 0; i < data.length; i++) {
            Map<String, String> features = new HashMap();
            for (int j = 0; j < colName.length - 1; j++) {
                features.put(colName[j], data[i][j]);
            }
            bayes.learn(data[i][colName.length - 1], features);
        }
        Map<String, String> features = new HashMap();
        features.put("outlook", "Overcast");
        features.put("temp", "Cool");
        features.put("humidity", "High");
        features.put("wind", "Strong");
        IClassification predict = bayes.classify(features, true);
        assertNotNull(predict);
        assertEquals(predict.getClassProbabilities().length, 2);
        assertEquals(predict.getClassProbabilities()[0].getCategory(), "Yes");
        assertEquals(predict.getClassProbabilities()[1].getCategory(), "No");
        assertEquals(predict.getClassProbabilities()[0].getProbability(), 0.7215830648872527, .0001);
        assertEquals(predict.getClassProbabilities()[1].getProbability(), 0.2784169351127473, .0001);
        bayes.dbCloseAndDestroy();
    }

    /**
     * Test based on
     * https://towardsdatascience.com/introduction-to-na%C3%AFve-bayes-classifier-fa59e3e24aaf
     */
    @Test
    public void testLearnClassifySample2() throws Exception {
        String[] cats = {ZERO, ONE};
        // Create a new bayes classifier with string categories and string features.
        // INaiveBayesClassifier bayes1 = new NaiveBayesClassifierLevelDBImpl("sentiment", cats, ".", 100);
        NaiveBayesClassifierMapLaplacedImpl bayes = new NaiveBayesClassifierMapLaplacedImpl("sentiment", cats, 1, true);
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
        IClassification predict = bayes.classify(features, true);
        assertNotNull(predict);
        assertEquals(predict.getClassProbabilities().length, 2);
        assertEquals(predict.getClassProbabilities()[0].getCategory(), "0");
        assertEquals(predict.getClassProbabilities()[1].getCategory(), "1");
        assertEquals(predict.getClassProbabilities()[0].getProbability(), 0.6511627906976744, .0001);
        assertEquals(predict.getClassProbabilities()[1].getProbability(), 0.3488372093023256, .0001);
    }


    /**
     * Test based on
     * https://towardsdatascience.com/introduction-to-na%C3%AFve-bayes-classifier-fa59e3e24aaf
     */
    @Test
    public void testLearnClassifyMapDBSample2() throws Exception {
        String[] cats = {ZERO, ONE};
        // Create a new bayes classifier with string categories and string features.
        // INaiveBayesClassifier bayes1 = new NaiveBayesClassifierLevelDBImpl("sentiment", cats, ".", 100);
        NaiveBayesClassifierMapLaplacedImpl bayes = new NaiveBayesClassifierMapLaplacedImpl("sentiment", cats, 1, true, MAPDB_DIR + System.currentTimeMillis());
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
        IClassification predict = bayes.classify(features, true);
        assertNotNull(predict);
        assertEquals(predict.getClassProbabilities().length, 2);
        assertEquals(predict.getClassProbabilities()[0].getCategory(), "0");
        assertEquals(predict.getClassProbabilities()[1].getCategory(), "1");
        assertEquals(predict.getClassProbabilities()[0].getProbability(), 0.6511627906976744, .0001);
        assertEquals(predict.getClassProbabilities()[1].getProbability(), 0.3488372093023256, .0001);
        bayes.dbCloseAndDestroy();
    }
}
