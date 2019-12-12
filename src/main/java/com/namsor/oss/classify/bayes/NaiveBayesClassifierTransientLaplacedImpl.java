package com.namsor.oss.classify.bayes;

import java.io.*;
import java.util.Arrays;

import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Naive Bayes Classifier with Laplace smoothing and implementation with
 * in-memory, concurrent ConcurrentHashMap. The Laplace smoothing has two
 * variants as per Sample1 and Sample2.
 *
 * @author elian carsenat, NamSor SAS
 */
public class NaiveBayesClassifierTransientLaplacedImpl extends AbstractNaiveBayesClassifierTransientImpl implements INaiveBayesClassifier {

    private static final boolean VARIANT = false;
    private static final double ALPHA = 1d;
    private final boolean variant;
    private final double alpha;

    public NaiveBayesClassifierTransientLaplacedImpl(String classifierName, String[] categories) throws IOException {
        this(classifierName, categories, ALPHA, VARIANT);
    }

    /**
     * Create a classifier
     *
     * @param classifierName
     * @param categories
     * @param alpha Typically 1
     * @param variant
     * @throws IOException
     */
    public NaiveBayesClassifierTransientLaplacedImpl(String classifierName, String[] categories, double alpha, boolean variant) throws IOException {
        super(classifierName, categories);
        this.alpha = alpha;
        this.variant = variant;
    }

    @Override
    public synchronized void learn(String category, Map<String, String> features, long weight) throws ClassifyException {
        String pathGlobal = pathGlobal();
        String pathGlobalCountCategories = pathGlobalCountCategories();
        getDb().put(pathGlobal, (getDb().containsKey(pathGlobal) ? getDb().get(pathGlobal) + weight : weight));
        String pathCategory = pathCategory(category);
        if (getDb().containsKey(pathCategory)) {
            getDb().put(pathCategory, getDb().get(pathCategory) + weight);
        } else {
            getDb().put(pathCategory, weight);
            // increment the count
            getDb().put(pathGlobalCountCategories, (getDb().containsKey(pathGlobalCountCategories) ? getDb().get(pathGlobalCountCategories) + 1 : 1));
        }
        for (Entry<String, String> feature : features.entrySet()) {
            String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
            getDb().put(pathCategoryFeatureKey, (getDb().containsKey(pathCategoryFeatureKey) ? getDb().get(pathCategoryFeatureKey) + weight : weight));
            String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
            String pathFeatureKeyValue = pathFeatureKeyValue(feature.getKey(), feature.getValue());
            if (getDb().containsKey(pathFeatureKeyValue)) {
                getDb().put(pathFeatureKeyValue, getDb().get(pathFeatureKeyValue) + weight);
            } else {
                getDb().put(pathFeatureKeyValue, weight);
                // increment the count
                String pathFeatureKeyCountValueTypes = pathFeatureKeyCountValueTypes(feature.getKey());
                getDb().put(pathFeatureKeyCountValueTypes, (getDb().containsKey(pathFeatureKeyCountValueTypes) ? getDb().get(pathFeatureKeyCountValueTypes) + 1 : 1));
            }
            getDb().put(pathCategoryFeatureKeyValue, (getDb().containsKey(pathCategoryFeatureKeyValue) ? getDb().get(pathCategoryFeatureKeyValue) + weight : weight));
        }
    }

    @Override
    public synchronized IClassification[] classify(Map<String, String> features) throws ClassifyException {
        String pathGlobal = pathGlobal();
        String pathGlobalCountCategories = pathGlobalCountCategories();
        long globalCount = (getDb().containsKey(pathGlobal) ? getDb().get(pathGlobal) : 0);
        long globalCountCategories = (getDb().containsKey(pathGlobalCountCategories) ? getDb().get(pathGlobalCountCategories) : 0);
        double[] likelyhood = new double[getCategories().length];
        double likelyhoodTot = 0;
        for (int i = 0; i < getCategories().length; i++) {
            String category = getCategories()[i];
            String pathCategory = pathCategory(category);
            long categoryCount = (getDb().containsKey(pathCategory) ? getDb().get(pathCategory) : 0);
            double product = 1.0d;
            for (Entry<String, String> feature : features.entrySet()) {
                String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                double featureCount = (getDb().containsKey(pathCategoryFeatureKey) ? getDb().get(pathCategoryFeatureKey) : 0);

                String pathFeatureKeyCountValueTypes = pathFeatureKeyCountValueTypes(feature.getKey());
                double featureCountValueTypes = (getDb().containsKey(pathFeatureKeyCountValueTypes) ? getDb().get(pathFeatureKeyCountValueTypes) : 0);

                String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                double featureCategoryCount = (getDb().containsKey(pathCategoryFeatureKeyValue) ? getDb().get(pathCategoryFeatureKeyValue) : 0);
                double basicProbability = (featureCount == 0 ? 0 : 1d * (featureCategoryCount + alpha) / (featureCount + featureCountValueTypes * alpha));
                product *= basicProbability;
            }
            if (variant) {
                likelyhood[i] = 1d * ((categoryCount + alpha) / (globalCount + globalCountCategories * alpha)) * product;
            } else {
                likelyhood[i] = 1d * categoryCount / globalCount * product;
            }
            likelyhoodTot += likelyhood[i];
        }
        return likelihoodsToProbas(likelyhood, likelyhoodTot);
    }

}
