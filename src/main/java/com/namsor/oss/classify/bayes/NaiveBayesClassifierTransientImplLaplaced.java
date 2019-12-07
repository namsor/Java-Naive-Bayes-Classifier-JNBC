/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

import java.io.*;
import java.util.Arrays;

import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Implementation : in-memory, using a concurrent ConcurrentHashMap
 *
 * @author elian carsenat, NamSor SAS
 */
public class NaiveBayesClassifierTransientImplLaplaced extends AbstractNaiveBayesClassifierImpl implements INaiveBayesClassifier {

    private final Map<String, Long> db;

    public NaiveBayesClassifierTransientImplLaplaced(String classifierName, String[] categories) throws IOException {
        super(classifierName, categories);
        db = new ConcurrentHashMap();
    }

    public String dbStatus() {
        return "OK";
    }

    public void dbClose() throws IOException {
    }

    public void dbCloseAndDestroy() throws IOException {
    }

    @Override
    public synchronized void learn(String category, Map<String, String> features, long weight) throws ClassifyException {
        //private Map<K, Map<T, Counter>> featureCountPerCategory;
        String pathGlobal = pathGlobal(); 
        db.put(pathGlobal, (db.containsKey(pathGlobal) ? db.get(pathGlobal) + weight : weight ));
        String pathCategory = pathCategory(category);
        db.put(pathCategory, (db.containsKey(pathCategory) ? db.get(pathCategory) + weight : weight ));
        for (Entry<String, String> feature : features.entrySet()) {
            String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
            db.put(pathCategoryFeatureKey, (db.containsKey(pathCategoryFeatureKey) ? db.get(pathCategoryFeatureKey) + weight : weight ));
            String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
            if( db.containsKey(pathCategoryFeatureKeyValue) ) {
                db.put(pathCategoryFeatureKeyValue, db.get(pathCategoryFeatureKeyValue) + weight);
            } else {
                db.put(pathCategoryFeatureKeyValue, weight);
                // increment the count
                String pathCategoryFeatureKeyValueCount = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue())
            }
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
                Logger.getLogger(NaiveBayesClassifierTransientImplLaplaced.class.getName()).log(Level.SEVERE, null, ex);
                throw new ClassifyException(ex);
            }
        }
    }
    
}
