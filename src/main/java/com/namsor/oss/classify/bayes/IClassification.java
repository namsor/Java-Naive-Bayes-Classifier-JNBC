package com.namsor.oss.classify.bayes;

/**
 * Classification output and probability estimate.
 * @author elian carsenat, NamSor SAS
 */
public interface IClassification {
    /**
     * Category
     * @return 
     */
    String getCategory();
    /**
     * Probability
     * @return 
     */
    double getProbability();
}
