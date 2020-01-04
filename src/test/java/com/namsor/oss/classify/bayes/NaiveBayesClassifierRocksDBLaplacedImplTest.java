package com.namsor.oss.classify.bayes;

import static com.namsor.oss.samples.MainSample1.NO;
import static com.namsor.oss.samples.MainSample1.YES;
import static com.namsor.oss.samples.MainSample1.colName;
import static com.namsor.oss.samples.MainSample1.data;
import static com.namsor.oss.samples.MainSample2.ONE;
import static com.namsor.oss.samples.MainSample2.X1;
import static com.namsor.oss.samples.MainSample2.X2;
import static com.namsor.oss.samples.MainSample2.Y;
import static com.namsor.oss.samples.MainSample2.ZERO;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * todo remove comment
 * @author elian
 */
public class NaiveBayesClassifierRocksDBLaplacedImplTest {
    private static final String ROCKSDB_DIR = "/tmp/rocksdb";
    public NaiveBayesClassifierRocksDBLaplacedImplTest() {
    }

    // todo methods don't don anything
    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
        File rocksdb = new File(ROCKSDB_DIR);
        if( rocksdb.exists() && rocksdb.isDirectory() ) {
            // ok
        } else {
            rocksdb.mkdirs();
        }
    }

    @After
    public void tearDown() {
    }

    /**
     * Test based on https://taylanbil.github.io/boostedNB or
     * http://ai.fon.bg.ac.rs/wp-content/uploads/2015/04/ML-Classification-NaiveBayes-2014.pdf
     */
    @Test
    public void testLearnClassifySample1() throws Exception {
        String[] cats = {YES, NO};
        NaiveBayesClassifierRocksDBLaplacedImpl bayes = new NaiveBayesClassifierRocksDBLaplacedImpl("tennis", cats, ROCKSDB_DIR, 1d, false);
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
        IClassification predict = bayes.classify(features,true);
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
        NaiveBayesClassifierRocksDBLaplacedImpl bayes = new NaiveBayesClassifierRocksDBLaplacedImpl("sentiment", cats, ROCKSDB_DIR, 1, true);
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
        IClassification predict = bayes.classify(features,true);
        assertNotNull(predict);
        assertEquals(predict.getClassProbabilities().length, 2);
        assertEquals(predict.getClassProbabilities()[0].getCategory(), "0");
        assertEquals(predict.getClassProbabilities()[1].getCategory(), "1");
        assertEquals(predict.getClassProbabilities()[0].getProbability(), 0.6511627906976744, .0001);
        assertEquals(predict.getClassProbabilities()[1].getProbability(), 0.3488372093023256, .0001);
        bayes.dbCloseAndDestroy();
        
    }

}
