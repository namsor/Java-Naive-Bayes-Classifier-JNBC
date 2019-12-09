package com.namsor.oss.classify.bayes;

/**
 * Classification output and probability estimate.
 * @author elian carsenat, NamSor SAS
 */
public interface IClassification {
    /**
     * Category
     * @return The classification category
     */
    String getCategory();
    /**
     * Probability
     * @return The classification probability estimate
     */
    double getProbability();
}
