/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

import java.util.Comparator;
import java.util.Set;

/**
 * A simple, scalable Naive Bayes Classifier, based on a key-value store (in memory, or disk-based)
 * @author elian carsenat, NamSor SAS
 */
public abstract class AbstractNaiveBayesClassifierImpl implements INaiveBayesClassifier {

    /**
     * @return the classifierName
     */
    public String getClassifierName() {
        return classifierName;
    }
    private final String classifierName;

    public AbstractNaiveBayesClassifierImpl(String classifierName, String[] categories) {
        this.classifierName = classifierName;
        this.categories = categories;
    }

    private final String[] categories;

    /**
     * @return the categories
     */
    public String[] getCategories() {
        return categories;
    }
    protected static final String KEY_GLOBAL = "~gL";
    protected static final String KEY_CATEGORY = "~cA";
    protected static final String KEY_FEATURE = "~fE";
    protected static final String KEY_SEPARATOR = "//";
    protected final Comparator<IClassification> orderByProba = new Comparator() {
        @Override
        public int compare(Object o1, Object o2) {
            IClassification c1 = (IClassification) o1;
            IClassification c2 = (IClassification) o2;
            return ((Double) c2.getProbability()).compareTo((Double) c1.getProbability());
        }
    };

    @Override
    public synchronized void learn(String category, Set<String> features) throws ClassifyException {
        learn(category, features, 1);
    }
}
