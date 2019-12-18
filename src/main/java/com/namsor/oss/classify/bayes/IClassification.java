package com.namsor.oss.classify.bayes;

/**
 * Classification output and probability estimate.
 *
 * @author elian carsenat, NamSor SAS
 */
public interface IClassification {

    String SPECIAL_CATEGORY_OTHER = "#Other";

    /**
     * Category
     *
     * @return The classification category
     */
    String getCategory();

    /**
     * Probability
     *
     * @return The classification probability estimate
     */
    double getProbability();

    /**
     * A special category that represents Other categories with a probability
     * sum, appears last in classification results.
     *
     * @return
     */
    boolean isOther();
}
