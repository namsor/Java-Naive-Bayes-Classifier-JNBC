package com.namsor.oss.classify.bayes;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Naive Bayes Classifier with Laplace smoothing and implementation with
 * concurrent ConcurrentHashMap or persistent mapDB. The Laplace smoothing has two variants as per
 * Sample1 and Sample2.
 * Also, this implementation is using log(a*b)=log(a)+log(b) to reduce overflow risk
 * @author elian carsenat, NamSor SAS
 */
public class NaiveBayesClassifierMapLaplacedLogImpl extends NaiveBayesClassifierMapLaplacedImpl implements INaiveBayesClassifier {


    /**
     * Create in-memory classifier using ConcurrentHashMap and defaults for Laplace smoothing (ALPHA=1 and VARIANT=false)
     *
     * @param classifierName The classifier name
     * @param categories     The classification categories
     */
    public NaiveBayesClassifierMapLaplacedLogImpl(String classifierName, String[] categories) {
        super(classifierName, categories, ALPHA, VARIANT);
    }

    /**
     * Create a classifier with in-memory ConcurrentHashMap and Laplace parameters
     *
     * @param classifierName The classifier name
     * @param categories     The classification categories
     * @param alpha          The Laplace alpha, typically 1.0
     * @param variant        The Laplace variant
     */
    public NaiveBayesClassifierMapLaplacedLogImpl(String classifierName, String[] categories, double alpha, boolean variant) {
        super(classifierName, categories, alpha, variant);
    }


    /**
     * Create persistent classifier using org.mapdb.HTreeMap and Laplace parameters
     *
     * @param classifierName   The classifier name
     * @param categories       The classification categories
     * @param alpha            The Laplace alpha, typically 1.0
     * @param variant          The Laplace variant
     * @param rootPathWritable A writable directory for org.mapdb.HTreeMap storage
     */
    public NaiveBayesClassifierMapLaplacedLogImpl(String classifierName, String[] categories, double alpha, boolean variant, String rootPathWritable) {
        super(classifierName, categories, alpha, variant, rootPathWritable);
    }

    /**
     * Create persistent classifier using org.mapdb.HTreeMap and defaults for Laplace smoothing (ALPHA=1 and VARIANT=false)
     *
     * @param classifierName   The classifier name
     * @param categories       The classification categories
     * @param rootPathWritable A writable directory for org.mapdb.HTreeMap storage
     */
    public NaiveBayesClassifierMapLaplacedLogImpl(String classifierName, String[] categories, String rootPathWritable) {
        super(classifierName, categories, ALPHA, VARIANT, rootPathWritable);
    }


    @Override
    public IClassification classify(Map<String, String> features, final boolean explainData) throws ClassifyException {
        Map<String, Long> explanation = null;
        if (explainData) {
            explanation = new HashMap();
        }
        String pathGlobal = pathGlobal();
        String pathGlobalCountCategories = pathGlobalCountCategories();
        long globalCount = (getDb().containsKey(pathGlobal) ? getDb().get(pathGlobal) : 0);
        if (explainData) {
            explanation.put(pathGlobal, globalCount);
        }
        long globalCountCategories = (getDb().containsKey(pathGlobalCountCategories) ? getDb().get(pathGlobalCountCategories) : 0);
        if (explainData) {
            explanation.put(pathGlobalCountCategories, globalCountCategories);
        }
        double[] likelyhood = new double[getCategories().length];
        double likelyhoodTot = 0;
        for (int i = 0; i < getCategories().length; i++) {
            String category = getCategories()[i];
            String pathCategory = pathCategory(category);
            long categoryCount = (getDb().containsKey(pathCategory) ? getDb().get(pathCategory) : 0);
            if (explainData) {
                explanation.put(pathCategory, categoryCount);
            }
            double logProduct = 0.0d;
            for (Entry<String, String> feature : features.entrySet()) {
                String pathFeatureKey = pathFeatureKey(feature.getKey());
                long featureCount = (getDb().containsKey(pathFeatureKey) ? getDb().get(pathFeatureKey) : 0);
                if (explainData) {
                    explanation.put(pathFeatureKey, featureCount);
                }
                if (featureCount > 0) {
                    String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                    long categoryFeatureCount = (getDb().containsKey(pathCategoryFeatureKey) ? getDb().get(pathCategoryFeatureKey) : 0);
                    if (explainData) {
                        explanation.put(pathCategoryFeatureKey, categoryFeatureCount);
                    }
                    String pathFeatureKeyCountValueTypes = pathFeatureKeyCountValueTypes(feature.getKey());
                    long featureCountValueTypes = (getDb().containsKey(pathFeatureKeyCountValueTypes) ? getDb().get(pathFeatureKeyCountValueTypes) : 0);
                    if (explainData) {
                        explanation.put(pathFeatureKeyCountValueTypes, featureCountValueTypes);
                    }
                    String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                    long categoryFeatureValueCount = (getDb().containsKey(pathCategoryFeatureKeyValue) ? getDb().get(pathCategoryFeatureKeyValue) : 0);
                    if (explainData) {
                        explanation.put(pathCategoryFeatureKeyValue, categoryFeatureValueCount);
                    }
                    double basicLogProbability = (categoryFeatureCount == 0 ? 0 : 0d + Math.log(categoryFeatureValueCount + getAlpha()) - Math.log(categoryFeatureCount + featureCountValueTypes * getAlpha()));
                    logProduct += basicLogProbability;
                }
            }
            if (isVariant()) {
                likelyhood[i] = Math.exp( 0d + Math.log(categoryCount + getAlpha()) - Math.log((globalCount + globalCountCategories * getAlpha())) + logProduct );
            } else {
                likelyhood[i] = Math.exp( 0d + Math.log(categoryCount) - Math.log(globalCount) + logProduct);
            }
            likelyhoodTot += likelyhood[i];
        }
        return new ClassificationImpl(features, likelihoodsToProbas(likelyhood, likelyhoodTot), explanation, true, isVariant(), getAlpha(), likelyhoodTot, true);
    }

}
