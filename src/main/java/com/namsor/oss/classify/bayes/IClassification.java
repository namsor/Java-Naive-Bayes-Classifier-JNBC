package com.namsor.oss.classify.bayes;

/**
 * Classification output
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
