# Java-Naive-Bayes-Classifier-JNBC
A Java Naive Bayes Classifier that works on RocksDB or other key-value store. The interface and sample are inspired by Philipp Nolte's https://github.com/ptnplanet/Java-Naive-Bayes-Classifier/, but the implementation is a full rewrite to store the key/values into a ConcurrentHashMap(), LevelDB, LMDB, Redis or other. LevelDB is a fast key-value storage library written at Google that provides an ordered mapping from string keys to string values.

Maven Quick-Start
------------------

This Java Naive Bayes Classifier on RocksDB can be installed via the jitpack repository. Make sure to add it to your buildfile first.

```xml
<repositories>
  <repository>
    <id>jitpack.io</id>
    <url>https://jitpack.io</url>
  </repository>
</repositories>
```

Then, treat it as any other dependency.

```xml
<dependency>
    <groupId>com.github.namsor</groupId>
    <artifactId>Java-Naive-Bayes-LevelDB</artifactId>
    <version>v1.0.0</version>
</dependency>
```

For other build-tools (e.g. gradle), visit https://jitpack.io for configuration snippets.

Example
------------------

Here is an excerpt from the example http://ai.fon.bg.ac.rs/wp-content/uploads/2015/04/ML-Classification-NaiveBayes-2014.pdf. 

```java

package com.namsor.oss.classify.bayes;

import java.io.IOException;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Simple test inspired by
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
            NaiveBayesClassifierTransientImpl bayes = new NaiveBayesClassifierTransientImpl("tennis", cats);
            //NaiveBayesClassifierRocksDBImpl bayes = new NaiveBayesClassifierRocksDBImpl("intro", cats, ".", 100);

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
            features.put("outlook","Sunny");
            features.put("temp","Cool");
            features.put("humidity","High");
            features.put("wind","Strong");

            IClassification[] predict = bayes.classify(features);
            StringWriter sw = new StringWriter();
            bayes.dumpDb(sw);
            System.out.println(sw);
            for (int i = 0; i < predict.length; i++) {
                System.out.println("P(" + predict[i].getCategory() + ")=" + predict[i].getProbability());
            }
        } catch (PersistentClassifierException ex) {
            Logger.getLogger(MainSample1.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ClassifyException ex) {
            Logger.getLogger(MainSample1.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(MainSample1.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}



```

The GNU LGPLv3 License
------------------
Copyright (c) 2018 - Elian Carsenat, NamSor SAS
https://www.gnu.org/licenses/lgpl-3.0.en.html
