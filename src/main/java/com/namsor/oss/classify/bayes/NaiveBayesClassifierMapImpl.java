package com.namsor.oss.classify.bayes;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

/**
 * A simple, scalable Naive Bayes Classifier, based on a key-value store (in
 * memory using ConcurrentHashMap, or disk-based using org.mapdb.HTreeMap)
 *
 * @author elian
 */
public class NaiveBayesClassifierMapImpl extends AbstractNaiveBayesClassifierMapImpl implements INaiveBayesClassifier {

    /**
     * Create in-memory classifier using ConcurrentHashMap
     *
     * @param classifierName The classifier name
     * @param categories     The classification categories
     */
    public NaiveBayesClassifierMapImpl(String classifierName, String[] categories) {
        super(classifierName, categories);
    }

    /**
     * Create persistent classifier using org.mapdb.HTreeMap
     *
     * @param classifierName   The classifier name
     * @param categories       The classification categories
     * @param rootPathWritable A writable directory for org.mapdb.HTreeMap storage
     */
    public NaiveBayesClassifierMapImpl(String classifierName, String[] categories, String rootPathWritable) {
        super(classifierName, categories, rootPathWritable);
    }

    @Override
    public synchronized void learn(String category, Map<String, String> features, long weight) throws ClassifyException {
        String pathGlobal = pathGlobal();
        getDb().put(pathGlobal, (getDb().containsKey(pathGlobal) ? getDb().get(pathGlobal) + weight : weight));
        String pathCategory = pathCategory(category);
        getDb().put(pathCategory, (getDb().containsKey(pathCategory) ? getDb().get(pathCategory) + weight : weight));
        for (Entry<String, String> feature : features.entrySet()) {
            String pathFeatureKey = pathFeatureKey(feature.getKey());
            getDb().put(pathFeatureKey, (getDb().containsKey(pathFeatureKey) ? getDb().get(pathFeatureKey) + weight : weight));
            String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
            getDb().put(pathCategoryFeatureKey, (getDb().containsKey(pathCategoryFeatureKey) ? getDb().get(pathCategoryFeatureKey) + weight : weight));
            String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
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
        long globalCount = (getDb().containsKey(pathGlobal) ? getDb().get(pathGlobal) : 0);
        if (explainData) {
            explanation.put(pathGlobal, globalCount);
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
                    String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                    long categoryFeatureValueCount = (getDb().containsKey(pathCategoryFeatureKeyValue) ? getDb().get(pathCategoryFeatureKeyValue) : 0);
                    if (explainData) {
                        explanation.put(pathCategoryFeatureKeyValue, categoryFeatureValueCount);
                    }
                    double basicProbability = (categoryFeatureCount == 0 ? 0 : 1d * categoryFeatureValueCount / categoryFeatureCount);
                    product *= basicProbability;
                }
            }
            likelyhood[i] = 1d * categoryCount / globalCount * product;
            likelyhoodTot += likelyhood[i];
        }
        return new ClassificationImpl(features, likelihoodsToProbas(likelyhood, likelyhoodTot), explanation, likelyhoodTot);
    }

}
