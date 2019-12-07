package com.namsor.oss.classify.bayes;

import java.util.Comparator;
import java.util.Map;

/**
 * A simple, scalable Naive Bayes Classifier, based on a key-value store (in
 * memory, or disk-based)
 *
 * @author elian carsenat, NamSor SAS
 */
public abstract class AbstractNaiveBayesClassifierImpl implements INaiveBayesClassifier {
    private static final String KEY_GLOBAL = "~gL";
    private static final String KEY_CATEGORY = "~cA";
    private static final String KEY_SUM = ".sum";
    private static final String KEY_COUNT = ".count";
    private static final String KEY_FEATURE = "~fE";
    private static final String KEY_FEATURE_EQVAL = "=";
    private static final String KEY_SEPARATOR = "//";
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
    @Override
    public String[] getCategories() {
        return categories;
    }   
    
    protected final Comparator<IClassification> orderByProba = new Comparator() {
        @Override
        public int compare(Object o1, Object o2) {
            IClassification c1 = (IClassification) o1;
            IClassification c2 = (IClassification) o2;
            return ((Double) c2.getProbability()).compareTo((Double) c1.getProbability());
        }
    };

    @Override
    public synchronized void learn(String category, Map<String, String> features) throws ClassifyException {
        learn(category, features, 1);
    }
    

    /**
     * Path to the total number of observations
     * @return 
     */
    protected static String pathGlobal() {
        return KEY_GLOBAL;
    }
    
    /**
     * Path to the number of observations in a category
     * @param category
     * @return 
     */
    protected static String pathCategory(String category) {
        return KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category;
    }

    /**
     * Path to the number of observations in a category, with feature featureKey
     * @param category
     * @return 
     */
    protected static String pathCategoryFeatureKey(String category, String featureKey) {
        return KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + featureKey;
    }    

    /**
     * Path to the number of observations in a category, with feature featureKey and value featureValue
     * @param category
     * @return 
     */
    protected static String pathCategoryFeatureKeyValue(String category, String featureKey, String featureValue) {
        return KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + featureKey+KEY_FEATURE_EQVAL+featureValue;
    }    
}
