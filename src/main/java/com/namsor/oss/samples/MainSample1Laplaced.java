package com.namsor.oss.samples;

import com.namsor.oss.classify.bayes.ClassifyException;
import com.namsor.oss.classify.bayes.NaiveBayesClassifierMapLaplacedImpl;
import com.namsor.oss.classify.bayes.PersistentClassifierException;
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
 * Simple example of Naive Bayes Classification (Sport / No Sport) with Laplace smoothing inspired by
 * http://ai.fon.bg.ac.rs/wp-content/uploads/2015/04/ML-Classification-NaiveBayes-2014.pdf
 *
 * @author elian
 */
public class MainSample1Laplaced {

    public static final String YES = "Yes";
    public static final String NO = "No";
    /**
     * Header table as per https://taylanbil.github.io/boostedNB or
     * http://ai.fon.bg.ac.rs/wp-content/uploads/2015/04/ML-Classification-NaiveBayes-2014.pdf
     */
    public static final String[] colName = {
        "outlook", "temp", "humidity", "wind", "play"
    };

    /**
     * Data table as per https://taylanbil.github.io/boostedNB or
     * http://ai.fon.bg.ac.rs/wp-content/uploads/2015/04/ML-Classification-NaiveBayes-2014.pdf
     */
    public static final String[][] data = {
        {"Sunny", "Hot", "High", "Weak", "No"},
        {"Sunny", "Hot", "High", "Strong", "No"},
        {"Overcast", "Hot", "High", "Weak", "Yes"},
        {"Rain", "Mild", "High", "Weak", "Yes"},
        {"Rain", "Cool", "Normal", "Weak", "Yes"},
        {"Rain", "Cool", "Normal", "Strong", "No"},
        {"Overcast", "Cool", "Normal", "Strong", "Yes"},
        {"Sunny", "Mild", "High", "Weak", "No"},
        {"Sunny", "Cool", "Normal", "Weak", "Yes"},
        {"Rain", "Mild", "Normal", "Weak", "Yes"},
        {"Sunny", "Mild", "Normal", "Strong", "Yes"},
        {"Overcast", "Mild", "High", "Strong", "Yes"},
        {"Overcast", "Hot", "Normal", "Weak", "Yes"},
        {"Rain", "Mild", "High", "Strong", "No"},};

    
    
    public static final void main(String[] args) {
        
        try {
            String[] cats = {YES, NO};
            // Create a new bayes classifier with string categories and string features.
            //NaiveBayesClassifierTransientLaplacedImpl bayes = new NaiveBayesClassifierTransientLaplacedImpl ("tennis", cats, 1, false);
            //NaiveBayesClassifierRocksDBLaplacedImpl bayes = new NaiveBayesClassifierRocksDBLaplacedImpl("tennis", cats, ".", 100);
            //NaiveBayesClassifierLevelDBLaplacedImpl bayes = new NaiveBayesClassifierLevelDBLaplacedImpl("tennis", cats, ".", 100);
            NaiveBayesClassifierMapLaplacedImpl bayes = new NaiveBayesClassifierMapLaplacedImpl ("tennis", cats, 1, false);

// Examples to learn from.
            for (int i = 0; i < data.length; i++) {
                Map<String,String> features = new HashMap();
                for (int j = 0; j < colName.length - 1; j++) {
                    features.put(colName[j], data[i][j]);
                }
                bayes.learn(data[i][colName.length - 1], features);
            }
            /**
             * Calculate the likelihood that: Outlook = sunny (0.22) Temperature
             * = cool (0.33) Humidity = high (0.33) Windy = true (0.33) Play =
             * yes (0.64)
             */

// Here are is X(B,S) to classify.
            Map<String,String> features = new HashMap();
            features.put("outlook","Overcast");
            features.put("temp","Cool");
            features.put("humidity","High");
            features.put("wind","Strong");

            IClassification predict = bayes.classify(features,true);
            for (int i = 0; i < predict.getClassProbabilities().length; i++) {
                System.out.println("P(" + predict.getClassProbabilities()[i].getCategory() + ")=" + predict.getClassProbabilities()[i].getProbability());
            }
            if( predict.getExplanationData()!=null) {
                NaiveBayesExplainerImpl explainer = new NaiveBayesExplainerImpl();
                IClassificationExplained explained = explainer.explain(predict);
                System.out.println(explained.toString());
                ScriptEngineManager scriptEngineManager = new ScriptEngineManager();
                ScriptEngine scriptEngine = scriptEngineManager.getEngineByName("JavaScript");
                // JavaScript code from String
                Object ob = scriptEngine.eval(explained.toString());
                System.out.println("Result of evaluating mathematical expressions in String = " + ob);
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
