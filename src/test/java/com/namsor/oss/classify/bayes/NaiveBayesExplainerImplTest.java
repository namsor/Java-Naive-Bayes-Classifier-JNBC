package com.namsor.oss.classify.bayes;

import com.namsor.oss.samples.MainSample1;
import static com.namsor.oss.samples.MainSample1.NO;
import com.namsor.oss.samples.MainSample3;

import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;

import static com.namsor.oss.samples.MainSample1.YES;
import static com.namsor.oss.samples.MainSample1.colName;
import static com.namsor.oss.samples.MainSample1.data;
import static com.namsor.oss.samples.MainSample2.*;
import static com.namsor.oss.samples.MainSample3.*;
import java.util.HashMap;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class NaiveBayesExplainerImplTest {

    public NaiveBayesExplainerImplTest() {
    }


    /**
     * Test based on https://taylanbil.github.io/boostedNB or
     * http://ai.fon.bg.ac.rs/wp-content/uploads/2015/04/ML-Classification-NaiveBayes-2014.pdf
     */
    @Test
    public void testLearnClassifyExplainSample1() throws Exception {
        String[] cats = {YES, NO};
        NaiveBayesClassifierMapImpl bayes = new NaiveBayesClassifierMapImpl("tennis", cats);
        for (int i = 0; i < MainSample1.data.length; i++) {
            Map<String, String> features = new HashMap();
            for (int j = 0; j < MainSample1.colName.length - 1; j++) {
                features.put(MainSample1.colName[j], MainSample1.data[i][j]);
            }
            bayes.learn(MainSample1.data[i][MainSample1.colName.length - 1], features);
        }
        Map<String, String> features = new HashMap();
        features.put("outlook", "Sunny");
        features.put("temp", "Cool");
        features.put("humidity", "High");
        features.put("wind", "Strong");
        IClassification predict = bayes.classify(features, true);
        assertNotNull(predict);
        assertEquals(predict.getClassProbabilities().length, 2);
        assertEquals(predict.getClassProbabilities()[0].getCategory(), "No");
        assertEquals(predict.getClassProbabilities()[1].getCategory(), "Yes");
        assertEquals(predict.getClassProbabilities()[0].getProbability(), 0.795417348608838, .0001);
        assertEquals(predict.getClassProbabilities()[1].getProbability(), 0.204582651391162, .0001);
        assertNotNull(predict.getExplanationData());

        NaiveBayesExplainerImpl explainer = new NaiveBayesExplainerImpl();
        IClassificationExplained explained = explainer.explain(predict);
        ScriptEngineManager scriptEngineManager = new ScriptEngineManager();
        ScriptEngine scriptEngine = scriptEngineManager.getEngineByName("JavaScript");
        // JavaScript code from String
        Double ob = (Double) scriptEngine.eval(explained.toJavaScriptText(features));
        assertEquals(ob, 0.795417348608838, .0001);
    }


    /**
     * Test based on
     * https://towardsdatascience.com/introduction-to-na%C3%AFve-bayes-classifier-fa59e3e24aaf
     */
    @Test
    public void testLearnClassifyExplainSample2() throws Exception {
        String[] cats = {ZERO, ONE};
        // Create a new bayes classifier with string categories and string features.
        // INaiveBayesClassifier bayes1 = new NaiveBayesClassifierLevelDBImpl("sentiment", cats, ".", 100);
        NaiveBayesClassifierMapImpl bayes = new NaiveBayesClassifierMapImpl("sentiment", cats);
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
        assertEquals(predict.getClassProbabilities()[0].getProbability(), 0.75, .0001);
        assertEquals(predict.getClassProbabilities()[1].getProbability(), 0.25, .0001);
        assertNotNull(predict.getExplanationData());

        NaiveBayesExplainerImpl explainer = new NaiveBayesExplainerImpl();
        IClassificationExplained explained = explainer.explain(predict);
        ScriptEngineManager scriptEngineManager = new ScriptEngineManager();
        ScriptEngine scriptEngine = scriptEngineManager.getEngineByName("JavaScript");
        // JavaScript code from String
        Double ob = (Double) scriptEngine.eval(explained.toJavaScriptText(features));
        assertEquals(ob, 0.75, .0001);
    }

    /**
     * Test based on
     * https://www.machinelearningplus.com/predictive-modeling/how-naive-bayes-algorithm-works-with-example-and-full-code/
     */
    @Test
    public void testLearnClassifyExplainSample3() throws Exception {
        String[] cats = {BANANA, ORANGE, OTHER};
        // Create a new bayes classifier with string categories and string features.
        NaiveBayesClassifierMapImpl bayes = new NaiveBayesClassifierMapImpl("fruit", cats);
        //NaiveBayesClassifierTransientLaplacedImpl bayes = new NaiveBayesClassifierTransientLaplacedImpl("fruit", cats);
        //NaiveBayesClassifierRocksDBImpl bayes = new NaiveBayesClassifierRocksDBImpl("intro", cats, ".", 100);

        // Examples to learn from.
        for (int i = 0; i < MainSample3.data.length; i++) {
            Map<String, String> features = new HashMap();
            features.put(MainSample3.colName[1], MainSample3.data[i][1]);
            features.put(MainSample3.colName[2], MainSample3.data[i][2]);
            features.put(MainSample3.colName[3], MainSample3.data[i][3]);
            bayes.learn(MainSample3.data[i][0], features, Long.parseLong(MainSample3.data[i][4]));
        }

        // Calculate the likelihood that: Long, Sweet, Yellow is a Banana
        // Here are is X(B,S) to classify.
        Map<String, String> features = new HashMap();
        features.put("Long", "Yes");
        features.put("Sweet", "Yes");
        features.put("Yellow", "Yes");
        features.put("Dummy", "Yes");
        IClassification predict = bayes.classify(features, true);
        assertNotNull(predict);
        assertEquals(predict.getClassProbabilities().length, 3);
        assertEquals(predict.getClassProbabilities()[0].getCategory(), BANANA);
        assertEquals(predict.getClassProbabilities()[1].getCategory(), OTHER);
        assertEquals(predict.getClassProbabilities()[2].getCategory(), ORANGE);
        assertEquals(predict.getClassProbabilities()[0].getProbability(), 0.9307479224376731, .0001);
        assertEquals(predict.getClassProbabilities()[1].getProbability(), 0.06925207756232689, .0001);
        assertEquals(predict.getClassProbabilities()[2].getProbability(), 0, .0001);
        assertNotNull(predict.getExplanationData());

        NaiveBayesExplainerImpl explainer = new NaiveBayesExplainerImpl();
        IClassificationExplained explained = explainer.explain(predict);
        ScriptEngineManager scriptEngineManager = new ScriptEngineManager();
        ScriptEngine scriptEngine = scriptEngineManager.getEngineByName("JavaScript");
        // JavaScript code from String
        Double ob = (Double) scriptEngine.eval(explained.toJavaScriptText(features));
        assertEquals(ob, 0.9307479224376731, .0001);

    }


    
    /**
     * Test based on https://taylanbil.github.io/boostedNB or
     * http://ai.fon.bg.ac.rs/wp-content/uploads/2015/04/ML-Classification-NaiveBayes-2014.pdf
     */
    @Test
    public void testLearnClassifyExplainLaplacedSample1() throws Exception {
        String[] cats = {YES, NO};
        NaiveBayesClassifierMapLaplacedImpl bayes = new NaiveBayesClassifierMapLaplacedImpl("tennis", cats);
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
        assertNotNull(predict.getExplanationData());

        NaiveBayesExplainerImpl explainer = new NaiveBayesExplainerImpl();
        IClassificationExplained explained = explainer.explain(predict);
        ScriptEngineManager scriptEngineManager = new ScriptEngineManager();
        ScriptEngine scriptEngine = scriptEngineManager.getEngineByName("JavaScript");
        // JavaScript code from String
        Double ob = (Double) scriptEngine.eval(explained.toJavaScriptText(features));
        assertEquals(ob,  0.7215830648872527, .0001);
    }


    /**
     * Test based on
     * https://towardsdatascience.com/introduction-to-na%C3%AFve-bayes-classifier-fa59e3e24aaf
     */
    @Test
    public void testLearnClassifyExplainLaplacedSample2() throws Exception {
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

        NaiveBayesExplainerImpl explainer = new NaiveBayesExplainerImpl();
        IClassificationExplained explained = explainer.explain(predict);
        ScriptEngineManager scriptEngineManager = new ScriptEngineManager();
        ScriptEngine scriptEngine = scriptEngineManager.getEngineByName("JavaScript");
        // JavaScript code from String
        Double ob = (Double) scriptEngine.eval(explained.toJavaScriptText(features));
        assertEquals(ob, 0.6511627906976744, .0001);
    }



    
    /**
     * Test based on https://taylanbil.github.io/boostedNB or
     * http://ai.fon.bg.ac.rs/wp-content/uploads/2015/04/ML-Classification-NaiveBayes-2014.pdf
     */
    @Test
    public void testLearnClassifyExplainLaplacedLogSample1() throws Exception {
        String[] cats = {YES, NO};
        NaiveBayesClassifierMapLaplacedLogImpl bayes = new NaiveBayesClassifierMapLaplacedLogImpl("tennis", cats);
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
        assertNotNull(predict.getExplanationData());

        NaiveBayesExplainerImpl explainer = new NaiveBayesExplainerImpl();
        IClassificationExplained explained = explainer.explain(predict);
        ScriptEngineManager scriptEngineManager = new ScriptEngineManager();
        ScriptEngine scriptEngine = scriptEngineManager.getEngineByName("JavaScript");
        // JavaScript code from String
        Double ob = (Double) scriptEngine.eval(explained.toJavaScriptText(features));
        assertEquals(ob,  0.7215830648872527, .0001);
    }


    /**
     * Test based on
     * https://towardsdatascience.com/introduction-to-na%C3%AFve-bayes-classifier-fa59e3e24aaf
     */
    @Test
    public void testLearnClassifyExplainLaplacedLogSample2() throws Exception {
        String[] cats = {ZERO, ONE};
        // Create a new bayes classifier with string categories and string features.
        // INaiveBayesClassifier bayes1 = new NaiveBayesClassifierLevelDBImpl("sentiment", cats, ".", 100);
        NaiveBayesClassifierMapLaplacedLogImpl bayes = new NaiveBayesClassifierMapLaplacedLogImpl("sentiment", cats, 1, true);
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

        NaiveBayesExplainerImpl explainer = new NaiveBayesExplainerImpl();
        IClassificationExplained explained = explainer.explain(predict);
        ScriptEngineManager scriptEngineManager = new ScriptEngineManager();
        ScriptEngine scriptEngine = scriptEngineManager.getEngineByName("JavaScript");
        // JavaScript code from String
        Double ob = (Double) scriptEngine.eval(explained.toJavaScriptText(features));
        assertEquals(ob, 0.6511627906976744, .0001);
    }

    
}
