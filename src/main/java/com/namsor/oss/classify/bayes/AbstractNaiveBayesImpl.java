/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

import java.util.Arrays;

/**
 * Functions common to the Naive Bayes Classifier and the Explainer
 * @author elian
 */
public class AbstractNaiveBayesImpl {
    
    private static final String KEY_GLOBAL = "gL";
    private static final String KEY_CATEGORY = "cA";
    private static final String KEY_COUNT = "_count";
    private static final String KEY_FEATURE = "fE";
    private static final String KEY_FEATURE_EQUAL = "_is_";
    private static final String KEY_SEPARATOR = "_";

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
        return KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + featureKey + KEY_FEATURE_EQUAL + featureValue;
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
        return KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + featureKey + KEY_FEATURE_EQUAL + featureValue;
    }
    

}
