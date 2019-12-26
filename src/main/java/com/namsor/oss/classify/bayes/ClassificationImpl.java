/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

import java.util.Map;

/**
 *
 * @author elian
 */
public class ClassificationImpl implements IClassification {

    private final IClassProbability[] classProbabilities;
    private final Map<String, Long> explanation;

    public ClassificationImpl(IClassProbability[] classProbabilities, Map<String, Long> explanation) {
        this.classProbabilities = classProbabilities;
        this.explanation = explanation;
    }

    /**
     * @return the classProbabilities
     */
    public IClassProbability[] getClassProbabilities() {
        return classProbabilities;
    }

    /**
     * @return the explanation
     */
    public Map<String, Long> getExplanation() {
        return explanation;
    }
}
