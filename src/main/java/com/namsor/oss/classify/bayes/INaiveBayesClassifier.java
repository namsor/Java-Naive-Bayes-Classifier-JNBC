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
     * @throws com.namsor.oss.classify.bayes.ClassifyException The classification error and cause
     */
    void learn(String category, Map<String, String> features) throws ClassifyException;

    /**
     * Learn from features
     *
     * @param category The category
     * @param features The features
     * @param weight   The weight
     * @throws com.namsor.oss.classify.bayes.ClassifyException The classification error and cause
     */
    void learn(String category, Map<String, String> features, long weight) throws ClassifyException;

    /**
     * Predict most probable class, optionally returning the data needed for future explanation.
     *
     * @param features    The features
     * @param explainData If should return the data needed for future explanation
     * @return The most likely classes with probability and (optionally) the explanation
     * @throws ClassifyException The classification error and cause
     */
    IClassification classify(Map<String, String> features, boolean explainData) throws ClassifyException;

    /**
     * This classifier has an immutable list of categories
     *
     * @return The classification categories
     */
    String[] getCategories();

    /**
     * Close the classifier (if persistent)
     *
     * @throws PersistentClassifierException The persistence error and cause
     */
    void dbClose() throws PersistentClassifierException;

    /**
     * Close the classifier (if persistent) and destroy the database.
     *
     * @throws com.namsor.oss.classify.bayes.PersistentClassifierException The persistence error and cause
     */
    void dbCloseAndDestroy() throws PersistentClassifierException;

    /**
     * Estimate the number of key-values in DB
     *
     * @return The estimate number of key values
     * @throws PersistentClassifierException The persistence error and cause
     */
    long dbSize() throws PersistentClassifierException;

    /**
     * Dump the current state of the model (can be large)
     *
     * @param w A writer
     * @throws PersistentClassifierException The persistence error and cause
     */
    void dumpDb(Writer w) throws PersistentClassifierException;
}
