package com.namsor.oss.classify.bayes;

import java.io.Writer;
import java.util.Map;

/**
 * Naive Bayes Classifier interface
 *
 * @author elian carsenat, NamSor SAS
 */
public interface INaiveBayesClassifier {

    /**
     * Learn from features
     *
     * @param category The category
     * @param features The features
     * @throws com.namsor.oss.classify.bayes.ClassifyException
     */
    void learn(String category, Map<String, String> features) throws ClassifyException;

    /**
     * Learn from features
     *
     * @param category The category
     * @param features The features
     * @param weight The weight
     * @throws com.namsor.oss.classify.bayes.ClassifyException
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
    
    /**
     * Estimate the number of key-values in DB
     * @return The estimate number of key values
     * @throws PersistentClassifierException 
     */
    long dbSize() throws PersistentClassifierException;
    
    /**
     * Dump the current state of the model (can be large)
     * @param w
     * @throws ClassifyException 
     */
    void dumpDb(Writer w) throws ClassifyException;
}
