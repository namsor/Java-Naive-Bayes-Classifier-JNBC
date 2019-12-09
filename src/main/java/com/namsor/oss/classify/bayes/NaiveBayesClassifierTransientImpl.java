package com.namsor.oss.classify.bayes;

import java.io.*;
import java.util.Arrays;

import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Naive Bayes Classifier implementation with in-memory, concurrent ConcurrentHashMap.
 *
 * @author elian carsenat, NamSor SAS
 */
public class NaiveBayesClassifierTransientImpl extends AbstractNaiveBayesClassifierImpl implements INaiveBayesClassifier {

    private final Map<String, Long> db;

    public NaiveBayesClassifierTransientImpl(String classifierName, String[] categories) throws IOException {
        super(classifierName, categories);
        db = new ConcurrentHashMap();
    }

    @Override
    public synchronized void learn(String category, Map<String, String> features, long weight) throws ClassifyException {
        String pathGlobal = pathGlobal(); 
        db.put(pathGlobal, (db.containsKey(pathGlobal) ? db.get(pathGlobal) + weight : weight ));
        String pathCategory = pathCategory(category);
        db.put(pathCategory, (db.containsKey(pathCategory) ? db.get(pathCategory) + weight : weight ));
        for (Entry<String, String> feature : features.entrySet()) {
            String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
            db.put(pathCategoryFeatureKey, (db.containsKey(pathCategoryFeatureKey) ? db.get(pathCategoryFeatureKey) + weight : weight ));
            String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
            db.put(pathCategoryFeatureKeyValue, (db.containsKey(pathCategoryFeatureKeyValue) ? db.get(pathCategoryFeatureKeyValue) + weight : weight ));
        }
    }

    @Override
    public synchronized IClassification[] classify(Map<String, String> features) throws ClassifyException {
        IClassification[] result = new ClassificationImpl[getCategories().length];
        String pathGlobal = pathGlobal(); 
        long globalCount = (db.containsKey(pathGlobal) ? db.get(pathGlobal) : 0);
        double[] likelyhood = new double[getCategories().length];
        double likelyhoodTot = 0;
        for (int i = 0; i < getCategories().length; i++) {
            String category = getCategories()[i];
            String pathCategory = pathCategory(category);
            long categoryCount = (db.containsKey(pathCategory) ? db.get(pathCategory) : 0);
            double product = 1.0d;
            for (Entry<String, String> feature : features.entrySet()) {
                String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                double featureCount = (db.containsKey(pathCategoryFeatureKey) ? db.get(pathCategoryFeatureKey) : 0);
                String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                double featureCategoryCount = (db.containsKey(pathCategoryFeatureKeyValue) ? db.get(pathCategoryFeatureKeyValue) : 0);
                double basicProbability = (featureCount == 0 ? 0 : 1d * featureCategoryCount / featureCount);
                product *= basicProbability;
            }
            likelyhood[i] = 1d * categoryCount / globalCount * product;
            likelyhoodTot += likelyhood[i];
        }
        for (int i = 0; i < getCategories().length; i++) {
            double proba = likelyhood[i] / likelyhoodTot;
            ClassificationImpl classif = new ClassificationImpl(getCategories()[i], proba); 
            result[i] = classif;
        }
        Arrays.sort(result, orderByProba);
        return result;
    }

    public synchronized void dumpDb(Writer w) throws ClassifyException {
        for (Map.Entry<String, Long> entry : db.entrySet()) {
            String key = entry.getKey();
            long value = entry.getValue();
            try {
                w.append(key + "|" + value + "\n");
            } catch (IOException ex) {
                Logger.getLogger(NaiveBayesClassifierTransientImpl.class.getName()).log(Level.SEVERE, null, ex);
                throw new ClassifyException(ex);
            }
        }
    }

    @Override
    public void dbCloseAndDestroy() throws PersistentClassifierException {
    }
    
}
