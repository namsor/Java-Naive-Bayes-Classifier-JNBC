package com.namsor.oss.classify.bayes;

import java.util.Map;

/**
 * Naive Bayes Classifier interface
 *
 * @author elian carsenat, NamSor SAS
 */
public interface INaiveBayesClassifier {

    /**
     * Learn
     *
     * @param category
     * @param features
     */
    void learn(String category, Map<String, String> features) throws ClassifyException;

    /**
     * Learn
     *
     * @param category
     * @param features
     */
    void learn(String category, Map<String, String> features, long weight) throws ClassifyException;

    /**
     * Predict most probable class
     *
     * @param features
     * @return
     */
    IClassification[] classify(Map<String, String> features) throws ClassifyException;

    /**
     * This classifier has an immutable list of categories
     *
     * @return
     */
    String[] getCategories();
    
    /**
     * Close the classifier (if persistent)
     * @throws PersistentClassifierException 
     */
    void dbClose() throws PersistentClassifierException;
    
    /**
     * Close the classifier (if persistent) and destroy the database.
     * @throws com.namsor.oss.classify.bayes.PersistentClassifierException
    */
    void dbCloseAndDestroy()  throws PersistentClassifierException;
}
