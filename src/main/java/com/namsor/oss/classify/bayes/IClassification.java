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
     * @return the classProbabilities : class name and probability (possibly the last class is 'Other')
     */
    IClassProbability[] getClassProbabilities();

    /**
     * @return the explanation : list of features and counts
     */
    Map<String, Long> getExplanation();

}
