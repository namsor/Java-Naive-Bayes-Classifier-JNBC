package com.namsor.oss.classify.bayes;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Naive Bayes Classifier with Laplace smoothing and implementation with
 * concurrent ConcurrentHashMap or persistent mapDB. The Laplace smoothing has two variants as per
 * Sample1 and Sample2.
 *
 * @author elian carsenat, NamSor SAS
 */
public class NaiveBayesClassifierMapLaplacedImpl extends AbstractNaiveBayesClassifierMapImpl implements INaiveBayesClassifier {

    private static final boolean VARIANT = false;
    private static final double ALPHA = 1d;
    private final boolean variant;
    private final double alpha;

    /**
     * Create in-memory classifier using ConcurrentHashMap and defaults for Laplace smoothing (ALPHA=1 and VARIANT=false)
     *
     * @param classifierName The classifier name
     * @param categories     The classification categories
     */
    public NaiveBayesClassifierMapLaplacedImpl(String classifierName, String[] categories) {
        this(classifierName, categories, ALPHA, VARIANT);
    }

    /**
     * Create a classifier with in-memory ConcurrentHashMap and Laplace parameters
     *
     * @param classifierName The classifier name
     * @param categories     The classification categories
     * @param alpha          The Laplace alpha, typically 1.0
     * @param variant        The Laplace variant
     */
    public NaiveBayesClassifierMapLaplacedImpl(String classifierName, String[] categories, double alpha, boolean variant) {
        super(classifierName, categories);
        this.alpha = alpha;
        this.variant = variant;
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
    public NaiveBayesClassifierMapLaplacedImpl(String classifierName, String[] categories, double alpha, boolean variant, String rootPathWritable) {
        super(classifierName, categories, rootPathWritable);
        this.alpha = alpha;
        this.variant = variant;
    }

    /**
     * Create persistent classifier using org.mapdb.HTreeMap and defaults for Laplace smoothing (ALPHA=1 and VARIANT=false)
     *
     * @param classifierName   The classifier name
     * @param categories       The classification categories
     * @param rootPathWritable A writable directory for org.mapdb.HTreeMap storage
     */
    public NaiveBayesClassifierMapLaplacedImpl(String classifierName, String[] categories, String rootPathWritable) {
        this(classifierName, categories, ALPHA, VARIANT, rootPathWritable);
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
            String pathFeatureKey = pathFeatureKey(feature.getKey());
            getDb().put(pathFeatureKey, (getDb().containsKey(pathFeatureKey) ? getDb().get(pathFeatureKey) + weight : weight));
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
            double product = 1.0d;
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
                    double basicProbability = (categoryFeatureCount == 0 ? 0 : 1d * (categoryFeatureValueCount + alpha) / (categoryFeatureCount + featureCountValueTypes * alpha));
                    product *= basicProbability;
                }
            }
            if (variant) {
                likelyhood[i] = 1d * ((categoryCount + alpha) / (globalCount + globalCountCategories * alpha)) * product;
            } else {
                likelyhood[i] = 1d * categoryCount / globalCount * product;
            }
            likelyhoodTot += likelyhood[i];
        }
        return new ClassificationImpl(features, likelihoodsToProbas(likelyhood, likelyhoodTot), explanation, true, variant, alpha, likelyhoodTot);
    }
}
