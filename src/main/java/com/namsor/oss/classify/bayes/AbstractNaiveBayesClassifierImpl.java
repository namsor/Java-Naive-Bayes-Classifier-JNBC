package com.namsor.oss.classify.bayes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * A simple, scalable Naive Bayes Classifier, based on a key-value store (in
 * memory, or disk-based)
 *
 * @author elian carsenat, NamSor SAS
 */
public abstract class AbstractNaiveBayesClassifierImpl implements INaiveBayesClassifier {
    private static final String P0 = "0";
    private static final String P1 = "1";
    private static final String P2 = "2";
    private static final String P3 = "3";
    private static final String P4 = "4";
    private static final String P5 = "5";
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

    protected final Comparator<IClassProbability> orderByProba = new Comparator() {
        @Override
        public int compare(Object o1, Object o2) {
            IClassProbability c1 = (IClassProbability) o1;
            IClassProbability c2 = (IClassProbability) o2;
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
        return P0+KEY_GLOBAL;
    }

    /**
     * Path to the total count of distinct categories
     *
     * @return Path to the total count of distinct categories
     */
    protected static String pathGlobalCountCategories() {
        return P0+KEY_GLOBAL+KEY_COUNT;
    }

    /**
     * Path to the number of observations in a category
     *
     * @param category The category
     * @return Path to the number of observations in a category
     */
    protected static String pathCategory(String category) {
        return P1+KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category;
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
        return P3+KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + featureKey;
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
        return P4+KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + featureKey + KEY_FEATURE_EQVAL + featureValue;
    }

    /**
     * Path to the counter for featureKey
     *
     * @param featureKey The featureKey
     * @return Path to the counter for feature featureKey
     */
    protected static String pathFeatureKey(String featureKey) {
        return P2+KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + featureKey;
    }

    /**
     * Path to the number of distinct value types for feature featureKey
     *
     * @param featureKey The featureKey
     * @return Path to the number of distinct value types for feature featureKey
     */
    protected static String pathFeatureKeyCountValueTypes(String featureKey) {
        return P2+KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + featureKey + KEY_COUNT;
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
        return P5+KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + featureKey + KEY_FEATURE_EQVAL + featureValue;
    }

    protected IClassProbability[] likelihoodsToProbas(double[] likelyhood, double likelyhoodTot) {
        if (topN > 0 && likelyhood.length > topN) {
            double[] likelyhoodSorted = likelyhood.clone(); // Arrays.copyOf(likelyhood, topN);
            Arrays.sort(likelyhoodSorted);
            double likelyhoodN = likelyhoodSorted[likelyhood.length - topN];
            List<IClassProbability> result = new ArrayList(topN + 1);
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
                    ClassProbabilityImpl classif = new ClassProbabilityImpl(getCategories()[i], proba);
                    result.add(classif);
                    probaTopN+=proba;
                }
            }
            Collections.sort(result, orderByProba);
            ClassProbabilityImpl classifOther = new ClassProbabilityImpl(IClassProbability.SPECIAL_CATEGORY_OTHER, (1d-probaTopN));
            result.add(classifOther);
            IClassProbability[] resultArr = new ClassProbabilityImpl[result.size()];
            resultArr = result.toArray(resultArr);
            return resultArr;
        } else {
            return likelihoodsToProbas_(likelyhood, likelyhoodTot);
        }
    }

    protected IClassProbability[] likelihoodsToProbas_(double[] likelyhood, double likelyhoodTot) {
        IClassProbability[] result = new ClassProbabilityImpl[getCategories().length];
        for (int i = 0; i < getCategories().length; i++) {
            double proba = likelyhood[i] / likelyhoodTot;
            if (proba > 1d) {
                // could equal 1.000000000002 due to double precision issue;
                proba = 1d;
            } else if (proba < 0) {
                proba = 0d;
            }
            ClassProbabilityImpl classif = new ClassProbabilityImpl(getCategories()[i], proba);
            result[i] = classif;
        }
        Arrays.sort(result, orderByProba);
        return result;
    }

}
