package com.namsor.oss.classify.bayes;

import java.util.Arrays;
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
    private final int topN;

    public AbstractNaiveBayesClassifierImpl(String classifierName, String[] categories) {
        this.classifierName = classifierName;
        this.categories = categories;
        this.topN = -1;
    }

    /**
     * Create a Naive Bayes Classifier that can return only the topN most
     * probable classifications, and a special 'Other'
     *
     * @param classifierName
     * @param categories
     * @param topN
     */
    public AbstractNaiveBayesClassifierImpl(String classifierName, String[] categories, int topN) {
        this.classifierName = classifierName;
        this.categories = categories;
        this.topN = topN;
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
     *
     * @return Path to the total number of observations
     */
    protected static String pathGlobal() {
        return KEY_GLOBAL;
    }

    /**
     * Path to the total count of distinct categories
     *
     * @return Path to the total count of distinct categories
     */
    protected static String pathGlobalCountCategories() {
        return KEY_GLOBAL + KEY_COUNT;
    }

    /**
     * Path to the number of observations in a category
     *
     * @param category The category
     * @return Path to the number of observations in a category
     */
    protected static String pathCategory(String category) {
        return KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category;
    }

    /**
     * Path to the number of observations in a category, with feature featureKey
     *
     * @param category The category
     * @param featureKey The feature key
     * @return Path to the number of observations in a category, with feature
     * featureKey
     */
    protected static String pathCategoryFeatureKey(String category, String featureKey) {
        return KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + featureKey;
    }

    /**
     * Path to the number of observations with feature featureKey and feature
     * value featureValue
     *
     * @param featureKey The feature key
     * @param featureValue The feature value
     * @return Path to the number of observations with feature featureKey and
     * feature value featureValue
     */
    protected static String pathFeatureKeyValue(String featureKey, String featureValue) {
        return KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + featureKey + KEY_FEATURE_EQVAL + featureValue;
    }

    /**
     * Path to the counter for featureKey
     *
     * @param featureKey The featureKey
     * @return Path to the counter for feature featureKey
     */
    protected static String pathFeatureKey(String featureKey) {
        return KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + featureKey;
    }

    /**
     * Path to the number of distinct value types for feature featureKey
     *
     * @param featureKey The featureKey
     * @return Path to the number of distinct value types for feature featureKey
     */
    protected static String pathFeatureKeyCountValueTypes(String featureKey) {
        return KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + featureKey + KEY_COUNT;
    }

    /**
     * Path to the number of observations in a category, with feature featureKey
     * and value featureValue
     *
     * @param category the Category
     * @param featureKey the featureKey
     * @param featureValue the featureValue
     * @return
     */
    protected static String pathCategoryFeatureKeyValue(String category, String featureKey, String featureValue) {
        return KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + featureKey + KEY_FEATURE_EQVAL + featureValue;
    }

    protected IClassification[] likelihoodsToProbas(double[] likelyhood, double likelyhoodTot) {
        if (topN > 0 && likelyhood.length > topN) {
            double[] likelyhoodSorted = likelyhood.clone(); // Arrays.copyOf(likelyhood, topN);
            Arrays.sort(likelyhoodSorted);
            double likelyhoodN = likelyhoodSorted[likelyhood.length - 1 - topN];
            IClassification[] result = new ClassificationImpl[topN + 1];
            double probaTopN = 0;
            for (int i = 0; i < getCategories().length; i++) {
                double proba = likelyhood[i] / likelyhoodTot;
                if (proba > 1d) {
                    // could equal 1.000000000002 due to double precision issue;
                    proba = 1d;
                } else if (proba < 0) {
                    proba = 0d;
                }
                if (likelyhood[i] >= likelyhoodN) {
                    ClassificationImpl classif = new ClassificationImpl(getCategories()[i], proba);
                    result[i] = classif;
                    probaTopN+=proba;
                }
            }
            Arrays.sort(result, 0, topN - 1, orderByProba);
            ClassificationImpl classifOther = new ClassificationImpl(IClassification.SPECIAL_CATEGORY_OTHER, (1d-probaTopN));
            result[topN] = classifOther;
            return result;
        } else {
            return likelihoodsToProbas_(likelyhood, likelyhoodTot);
        }
    }

    protected IClassification[] likelihoodsToProbas_(double[] likelyhood, double likelyhoodTot) {
        IClassification[] result = new ClassificationImpl[getCategories().length];
        for (int i = 0; i < getCategories().length; i++) {
            double proba = likelyhood[i] / likelyhoodTot;
            if (proba > 1d) {
                // could equal 1.000000000002 due to double precision issue;
                proba = 1d;
            } else if (proba < 0) {
                proba = 0d;
            }
            ClassificationImpl classif = new ClassificationImpl(getCategories()[i], proba);
            result[i] = classif;
        }
        Arrays.sort(result, orderByProba);
        return result;
    }

}
