/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

import java.util.Map;

/**
 * Classification output : class probabilities and (optionally) the features and counters for explanation / audit trail
 * @author elian
 */
public interface IClassification {
    
    /**
     * The ordered classes and probabilities.
     * @return the classProbabilities : class name and probability (possibly the last class is 'Other')
     */
    IClassProbability[] getClassProbabilities();

    /**
     * All the data needed to explain the results. 
     * @return the explanation : list of features and counts
     */
    Map<String, Long> getExplanationData();
    
    /**
     * Is Laplace smoothed? 
     * @return True if Laplace smoothed
     */
    boolean isLaplaceSmoothed();
    
    /**
     * If Laplace Smoothed With variant, then:  likelyhood[i] = 1d * ((categoryCount + alpha) / (globalCount + globalCountCategories * alpha)) * product;
     * otherwise: likelyhood[i] = 1d * categoryCount / globalCount * product;
     * @return 
     */
    boolean isLaplaceSmoothedVariant();

    /**
     * The input features
     * @return 
     */
    Map<String, String> getFeatures();
    
    /**
     * The alpha value used for Laplace smoothing
     * @return 
     */
    double getLaplaceSmoothingAlpha();
}
