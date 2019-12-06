# Java-Naive-Bayes-Classifier-JNBC
A Java Naive Bayes Classifier that works on LevelDB or other key-value store. The interface and sample are inspired by Philipp Nolte's https://github.com/ptnplanet/Java-Naive-Bayes-Classifier/, but the implementation is a full rewrite to store the key/values into a ConcurrentHashMap(), LevelDB, LMDB, Redis or other. LevelDB is a fast key-value storage library written at Google that provides an ordered mapping from string keys to string values.

Maven Quick-Start
------------------

This Java Naive Bayes Classifier on LevelDB can be installed via the jitpack repository. Make sure to add it to your buildfile first.

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

Here is an excerpt from the example (inspired by https://github.com/ptnplanet/Java-Naive-Bayes-Classifier/). The classifier will classify sentences (arrays of features) as sentences with either positive or negative sentiment. 

```java

            String[] cats = {POSITIVE, NEGATIVE};
            // Create a new bayes classifier with string categories and string features.
            INaiveBayesClassifier bayes = (USE_LEVELDB ? new NaiveBayesClassifierLevelDBImpl("sentiment", cats, ".", 100) : 
               new NaiveBayesClassifierTransientImpl("sentiment", cats));

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
            StringWriter sw = new StringWriter();
            
            //bayes.dumpDb(sw);
            //System.out.println(sw);
            System.out.println( // will output "positive"
            bayes.classify(new HashSet(Arrays.asList(unknownText1)))[0].getCategory());
            System.out.println( // will output "negative"
            bayes.classify(new HashSet(Arrays.asList(unknownText2)))[0].getCategory());


```

The GNU LGPLv3 License
------------------
Copyright (c) 2018 - Elian Carsenat, NamSor SAS
https://www.gnu.org/licenses/lgpl-3.0.en.html
