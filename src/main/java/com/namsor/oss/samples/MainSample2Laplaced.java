package com.namsor.oss.samples;

import com.namsor.oss.classify.bayes.ClassifyException;
import com.namsor.oss.classify.bayes.NaiveBayesClassifierMapLaplacedImpl;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import com.namsor.oss.classify.bayes.IClassification;
import com.namsor.oss.classify.bayes.IClassificationExplained;
import com.namsor.oss.classify.bayes.NaiveBayesExplainerImpl;
import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;

/**
 * Simple test inspired by
 * https://towardsdatascience.com/introduction-to-na%C3%AFve-bayes-classifier-fa59e3e24aaf
 *
 * @author elian
 */
public class MainSample2Laplaced {

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
            for (int i = 0; i < predict.getClassProbabilities().length; i++) {
                System.out.println("P(" + predict.getClassProbabilities()[i].getCategory() + ")=" + predict.getClassProbabilities()[i].getProbability());
            }
            if (predict.getExplanationData() != null) {
                NaiveBayesExplainerImpl explainer = new NaiveBayesExplainerImpl();
                IClassificationExplained explained = explainer.explain(predict);
                System.out.println(explained.toString());
                ScriptEngineManager scriptEngineManager = new ScriptEngineManager();
                ScriptEngine scriptEngine = scriptEngineManager.getEngineByName("JavaScript");
                // JavaScript code from String
                Object ob = scriptEngine.eval(explained.toString());
                System.out.println("Result of evaluating mathematical expressions in String = " + ob);
                
            }
        } catch (ClassifyException ex) {
            Logger.getLogger(MainSample2Laplaced.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Throwable ex) {
            Logger.getLogger(MainSample2Laplaced.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
