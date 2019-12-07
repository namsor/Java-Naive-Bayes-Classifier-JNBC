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
public class NaiveBayesClassifierTransientLaplacedImpl extends AbstractNaiveBayesClassifierImpl implements INaiveBayesClassifier {
    private static final boolean VARIANT=false;
    private static final double ALPHA=1d;
    private final boolean variant;
    private final double alpha;   
    private final Map<String, Long> db;
    
    public NaiveBayesClassifierTransientLaplacedImpl(String classifierName, String[] categories) throws IOException {
        super(classifierName, categories);
        db = new ConcurrentHashMap();
        this.alpha = ALPHA;
        this.variant = VARIANT;
    }
    /**
     * Create a classifier
     * @param classifierName
     * @param categories
     * @param alpha Typically 1
     * @param variant 
     * @throws IOException 
     */
    public NaiveBayesClassifierTransientLaplacedImpl(String classifierName, String[] categories, double alpha, boolean variant) throws IOException {
        super(classifierName, categories);
        db = new ConcurrentHashMap();
        this.alpha = alpha;
        this.variant = variant;
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
        String pathGlobalCountCategories = pathGlobalCountCategories();
        db.put(pathGlobal, (db.containsKey(pathGlobal) ? db.get(pathGlobal) + weight : weight ));
        String pathCategory = pathCategory(category);
        if( db.containsKey(pathCategory) ) {
            db.put(pathCategory, db.get(pathCategory) + weight );            
        } else {
            db.put(pathCategory, weight);
            // increment the count
            db.put(pathGlobalCountCategories, (db.containsKey(pathGlobalCountCategories) ? db.get(pathGlobalCountCategories) + 1 : 1 ));            
        }
        for (Entry<String, String> feature : features.entrySet()) {
            String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
            db.put(pathCategoryFeatureKey, (db.containsKey(pathCategoryFeatureKey) ? db.get(pathCategoryFeatureKey) + weight : weight ));
            String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
            String pathFeatureKeyValue = pathFeatureKeyValue(feature.getKey(), feature.getValue());
            if( db.containsKey(pathFeatureKeyValue)) {
                db.put(pathFeatureKeyValue, db.get(pathFeatureKeyValue) + weight);
            } else {
                db.put(pathFeatureKeyValue, weight);
                // increment the count
                String pathFeatureKeyCountValueTypes = pathFeatureKeyCountValueTypes(feature.getKey());
                db.put(pathFeatureKeyCountValueTypes, (db.containsKey(pathFeatureKeyCountValueTypes) ? db.get(pathFeatureKeyCountValueTypes) + 1 : 1 ));                
            }
            db.put(pathCategoryFeatureKeyValue, (db.containsKey(pathCategoryFeatureKeyValue) ? db.get(pathCategoryFeatureKeyValue) + weight : weight ));
        }
    }

    @Override
    public synchronized IClassification[] classify(Map<String, String> features) throws ClassifyException {
        IClassification[] result = new ClassificationImpl[getCategories().length];
        String pathGlobal = pathGlobal(); 
        String pathGlobalCountCategories = pathGlobalCountCategories();
        long globalCount = (db.containsKey(pathGlobal) ? db.get(pathGlobal) : 0);
        long globalCountCategories = (db.containsKey(pathGlobalCountCategories) ? db.get(pathGlobalCountCategories) : 0);
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

                String pathFeatureKeyCountValueTypes = pathFeatureKeyCountValueTypes(feature.getKey());
                double featureCountValueTypes = (db.containsKey(pathFeatureKeyCountValueTypes) ? db.get(pathFeatureKeyCountValueTypes) : 0);

                String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                double featureCategoryCount = (db.containsKey(pathCategoryFeatureKeyValue) ? db.get(pathCategoryFeatureKeyValue) : 0);
                double basicProbability = (featureCount == 0 ? 0 : 1d * (featureCategoryCount + alpha) / (featureCount + featureCountValueTypes * alpha) );
                product *= basicProbability;
            }
            if( variant ) {
                likelyhood[i] = 1d * ((categoryCount + alpha) / (globalCount + globalCountCategories * alpha)) * product;
            } else {
                likelyhood[i] = 1d * categoryCount / globalCount * product;
            }
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
                Logger.getLogger(NaiveBayesClassifierTransientLaplacedImpl.class.getName()).log(Level.SEVERE, null, ex);
                throw new ClassifyException(ex);
            }
        }
    }
    
}
