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
public class NaiveBayesClassifierTransientImpl extends AbstractNaiveBayesClassifierImpl implements INaiveBayesClassifier {

    private final Map<String, Long> db;

    public NaiveBayesClassifierTransientImpl(String classifierName, String[] categories) throws IOException {
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
    public synchronized void learn(String category, Map<String, String> features, int weight) throws ClassifyException {
        //private Map<K, Map<T, Counter>> featureCountPerCategory;
        db.put(bytes(pathGlobal()), Longs.toByteArray((db.get(bytes(pathGlobal())) == null ? weight : Longs.fromByteArray(db.get(bytes(pathGlobal()))) + weight)));
        db.put(bytes(pathCategory(category)), Longs.toByteArray((db.get(bytes(pathCategory(category))) == null ? weight : Longs.fromByteArray(db.get(bytes(pathCategory(category)))) + weight)));
        for (Entry<String, String> feature : features.entrySet()) {
            db.put(bytes(pathCategoryFeatureKey(category, feature.getKey())), Longs.toByteArray((db.get(bytes(pathCategoryFeatureKey(category, feature.getKey()))) == null ? weight : Longs.fromByteArray(db.get(bytes(pathCategoryFeatureKey(category, feature.getKey())))) + weight)));
            db.put(bytes(pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue())), Longs.toByteArray((db.get(bytes(pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue()))) == null ? weight : Longs.fromByteArray(db.get(bytes(pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue())))) + weight)));
        }
    }

    @Override
    public synchronized IClassification[] classify(Map<String, String> features) throws ClassifyException {
        IClassification[] result = new ClassificationImpl[getCategories().length];
        long globalCount = (db.get(bytes(pathGlobal())) == null ? 0 : Longs.fromByteArray(db.get(bytes(pathGlobal()))));
        double[] likelyhood = new double[getCategories().length];
        double likelyhoodTot = 0;
        for (int i = 0; i < getCategories().length; i++) {
            String category = getCategories()[i];
            long categoryCount = (db.get(bytes(pathCategory(category))) == null ? 0 : Longs.fromByteArray(db.get(bytes(pathCategory(category)))));
            double product = 1.0d;
            for (Entry<String, String> feature : features.entrySet()) {
                double featureCount = (db.get(bytes(pathCategoryFeatureKey(category, feature.getKey()))) == null ? 0 : Longs.fromByteArray(db.get(bytes(pathCategoryFeatureKey(category, feature.getKey())))));
                double featureCategoryCount = (db.get(bytes(pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue()))) == null ? 0 : Longs.fromByteArray(db.get(bytes(pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue())))));
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

    private String bytes(String key) {
        return key;
    }

    /**
     * This class just to have a very similar interface when using Transient HashMap vs. a persistent KeyValue store
     */
    private static class Longs {
        static long fromByteArray(Long get) {
            if (get == null) {
                return 0;
            } else {
                return get;
            }
        }
        static Long toByteArray(long l) {
            return l;
        }
    }

}
