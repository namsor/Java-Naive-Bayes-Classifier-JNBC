/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

import java.io.*;
import java.util.Arrays;
import java.util.Set;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Implementation : in-memory
 *
 * @author elian carsenat, NamSor SAS
 */
public class NaiveBayesClassifierTransientImpl extends AbstractNaiveBayesClassifierImpl implements INaiveBayesClassifier {

    private final Map<String, Long> db;

    public NaiveBayesClassifierTransientImpl(String classifierName, String[] categories) throws IOException {
        super(classifierName,categories);
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
    protected void finalize() throws Throwable {

    }

    @Override
    public synchronized void learn(String category, Set<String> features) throws ClassifyException {
        learn(category, features, 1);
    }

    @Override
    public synchronized void learn(String category, Set<String> features, int weight) throws ClassifyException {
        //private Map<K, Map<T, Counter>> featureCountPerCategory;
        db.put(bytes(KEY_GLOBAL), Longs.toByteArray((db.get(bytes(KEY_GLOBAL)) == null ? weight : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL))) + weight)));
        db.put(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category), Longs.toByteArray((db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category)) == null ? weight : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category))) + weight)));
        for (String feature : features) {
            db.put(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), Longs.toByteArray((db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature)) == null ? weight : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature))) + weight)));
            db.put(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), Longs.toByteArray((db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature)) == null ? weight : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature))) + weight)));
        }
    }

    @Override
    public synchronized void forget(String category, Set<String> features) throws ClassifyException {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public synchronized IClassification[] classify(Set<String> features) throws ClassifyException {
        IClassification[] result = new ClassificationImpl[getCategories().length];
        // return ((double) this.getCategoryCount(category) / (double) this.getCategoriesTotal()) * featuresProbabilityProduct(features, category);
        long globalCount = (db.get(bytes(KEY_GLOBAL)) == null ? 0 : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL))));
        for (int i = 0; i < getCategories().length; i++) {
            String category = getCategories()[i];
            long categoryCount = (db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category)) == null ? 0 : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category))));
            double product = 1.0d;
            double weight = 1.0d;
            double assumedProbability = 0.5d;
            for (String feature : features) {
                //product *= this.featureWeighedAverage(feature, category); //this.featureWeighedAverage(feature, category, null, 1.0d, 0.5d);
                double featureCount = (db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature)) == null ? 0 : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature))));
                double featureCategoryCount = (db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature)) == null ? 0 : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature))));
                double basicProbability = (featureCount == 0 ? 0 : 1d * featureCategoryCount / featureCount);

                double featureWeighedAverage = (weight * assumedProbability + featureCount * basicProbability) / (weight + featureCount);
                product *= featureWeighedAverage;
            }
            double proba = 1d * categoryCount / globalCount * product;
            IClassification classif = new ClassificationImpl(category, proba); // return ((double) this.getCategoryCount(category) / (double) this.getCategoriesTotal()) * featuresProbabilityProduct(features, category);
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
