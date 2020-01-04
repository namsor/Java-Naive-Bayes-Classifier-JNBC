package com.namsor.oss.classify.bayes;

/**
 * Classification output and probability estimate.
 *
 * @author elian carsenat, NamSor SAS
 */
public interface IClassProbability {

    /**
     * Category
     *
     * @return The classification category
     */
    String getCategory();

    /**
     * Probability
     *
     * @return he probability (likelyHood/sum(likelyhoods))
     */
    double getProbability();

}
