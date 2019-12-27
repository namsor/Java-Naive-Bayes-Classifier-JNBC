# Java-Naive-Bayes-Classifier-JNBC
A Java Naive Bayes Classifier that works in-memory or off the heap on fast key-value stores (MapDB, LevelDB or RocksDB). Naive Bayes Classification is fast. The objective of this ground-up implementations is to provide a self-contained, vertically scalable and explainable implementation.  

Maven Quick-Start
------------------

This Java Naive Bayes Classifier can be installed as any other dependency.

```xml
<dependency>
    <groupId>com.namsor</groupId>
    <artifactId>Java-Naive-Bayes-Classifier-JNBC</artifactId>
    <version>v2.0.4</version>
</dependency>
```

Example
------------------

Here is an excerpt from the example http://ai.fon.bg.ac.rs/wp-content/uploads/2015/04/ML-Classification-NaiveBayes-2014.pdf. 

```java

package com.namsor.oss.samples;

import com.namsor.oss.classify.bayes.ClassifyException;
import com.namsor.oss.classify.bayes.NaiveBayesClassifierMapImpl;
import com.namsor.oss.classify.bayes.PersistentClassifierException;
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
 * Simple example of Naive Bayes Classification (Sport / No Sport) inspired by
 * http://ai.fon.bg.ac.rs/wp-content/uploads/2015/04/ML-Classification-NaiveBayes-2014.pdf
 *
 * @author elian
 */
public class MainSample1 {

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
            NaiveBayesClassifierMapImpl bayes = new NaiveBayesClassifierMapImpl("tennis", cats);
            
            // Examples to learn from.
            for (int i = 0; i < data.length; i++) {
                Map<String, String> features = new HashMap();
                for (int j = 0; j < colName.length - 1; j++) {
                    features.put(colName[j], data[i][j]);
                }
                // learn ex. Category=Yes Conditions=Sunny, Cool, Normal and Weak.
                bayes.learn(data[i][colName.length - 1], features);
            }

            Map<String, String> features = new HashMap();
            features.put("outlook", "Sunny");
            features.put("temp", "Cool");
            features.put("humidity", "High");
            features.put("wind", "Strong");

            // Shall we play given weather conditions Sunny, Cool, Rainy and Windy ?
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
                Double proba = (Double) scriptEngine.eval(explained.toString());
                System.out.println("Result of evaluating mathematical expressions in String = " + proba);
            }
        } catch (PersistentClassifierException ex) {
            Logger.getLogger(MainSample1.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ClassifyException ex) {
            Logger.getLogger(MainSample1.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Throwable ex) {
            Logger.getLogger(MainSample1.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}

```
Explaining output 
------------------
When running the above example, we see that we are unlikely to play given the weather conditions Sunny, Cool, Rainy and Windy :
```
P(No)=0.795417348608838
P(Yes)=0.204582651391162
```
We can further explain how the likelyhoods were calculated by calling the Explainer. The explainer output can be humanly interpreted, but also the formulae and expressions can be interpreted using JavaScript. 

```
// observation table variables 
var gL=14
var gL_cA_No=5
var gL_cA_No_fE_humidity=5
var gL_cA_No_fE_humidity_is_High=4
var gL_cA_No_fE_outlook=5
var gL_cA_No_fE_outlook_is_Sunny=3
var gL_cA_No_fE_temp=5
var gL_cA_No_fE_temp_is_Cool=1
var gL_cA_No_fE_wind=5
var gL_cA_No_fE_wind_is_Strong=3
var gL_cA_Yes=9
var gL_cA_Yes_fE_humidity=9
var gL_cA_Yes_fE_humidity_is_High=3
var gL_cA_Yes_fE_outlook=9
var gL_cA_Yes_fE_outlook_is_Sunny=2
var gL_cA_Yes_fE_temp=9
var gL_cA_Yes_fE_temp_is_Cool=3
var gL_cA_Yes_fE_wind=9
var gL_cA_Yes_fE_wind_is_Strong=3
var gL_fE_humidity=14
var gL_fE_outlook=14
var gL_fE_temp=14
var gL_fE_wind=14


// likelyhoods by category 

// likelyhoods for category No
var likelyhoodOfNo=gL_cA_No / gL * (gL_cA_No_fE_temp_is_Cool / gL_cA_No_fE_temp * gL_cA_No_fE_humidity_is_High / gL_cA_No_fE_humidity * gL_cA_No_fE_outlook_is_Sunny / gL_cA_No_fE_outlook * gL_cA_No_fE_wind_is_Strong / gL_cA_No_fE_wind * 1 )
var likelyhoodOfNoExpr=5 / 14 * (1 / 5 * 4 / 5 * 3 / 5 * 3 / 5 * 1 )
var likelyhoodOfNoValue=0.020571428571428574

// likelyhoods for category Yes
var likelyhoodOfYes=gL_cA_Yes / gL * (gL_cA_Yes_fE_temp_is_Cool / gL_cA_Yes_fE_temp * gL_cA_Yes_fE_humidity_is_High / gL_cA_Yes_fE_humidity * gL_cA_Yes_fE_outlook_is_Sunny / gL_cA_Yes_fE_outlook * gL_cA_Yes_fE_wind_is_Strong / gL_cA_Yes_fE_wind * 1 )
var likelyhoodOfYesExpr=9 / 14 * (3 / 9 * 3 / 9 * 2 / 9 * 3 / 9 * 1 )
var likelyhoodOfYesValue=0.005291005291005291


// probability estimates by category 

// probability estimate for category No
var probabilityOfNo=likelyhoodOfNo/(likelyhoodOfNo+likelyhoodOfYes+0)
var probabilityOfNoValue=0.795417348608838

// probability estimate for category Yes
var probabilityOfYes=likelyhoodOfYes/(likelyhoodOfNo+likelyhoodOfYes+0)
var probabilityOfYesValue=0.204582651391162


// return the highest probability estimate for evaluation 
probabilityOfNo
```

Performance 
------------------
Binomial classifiers : the AbstractNaiveBayesClassifierMapImpl with in-memory ConcurrentHashMap can learn from billions of facts and classify new data very fast.
Using off-the-heap persistent key-value stores can help scaling vertically to even larger volumes. For example, the MapDB implementation on SSDs is only ~3-5 times slower and it can scale on large volumes. 

Multinomial classifiers : with many class categories and many features, you may need to use the in-memory ConcurrentHashMap implementation and allocate more memory to the java heap. This implementation is known to run smoothly on servers with 192Gb RAM. 
Further optimization will be needed to effectively use MemDB, LevelDB or RocksDB when the classification needs to read a LOT of data. 


The GNU LGPLv3 License
------------------
Copyright (c) 2018 - Elian Carsenat, NamSor SAS
https://www.gnu.org/licenses/lgpl-3.0.en.html
