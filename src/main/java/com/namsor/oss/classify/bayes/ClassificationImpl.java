/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

import java.io.StringWriter;
import java.util.Map;

/**
 *
 * @author elian
 */
public class ClassificationImpl implements IClassification {

    private final Map<String, String> features;
    private final IClassProbability[] classProbabilities;
    private final Map<String, Long> explanationData;
    private final boolean laplaceSmoothed;
    private final boolean laplaceSmoothedVariant;
    private final double laplaceSmoothingAlpha;

    public ClassificationImpl(Map<String, String> features, IClassProbability[] classProbabilities, Map<String, Long> explanationData, boolean laplaceSmoothed, boolean laplaceSmoothedVariant, double laplaceSmoothingAlpha) {
        this.features = features;
        this.classProbabilities = classProbabilities;
        this.explanationData = explanationData;
        this.laplaceSmoothed = laplaceSmoothed;
        this.laplaceSmoothedVariant = laplaceSmoothedVariant;
        this.laplaceSmoothingAlpha = laplaceSmoothingAlpha;
    }

    public ClassificationImpl(Map<String, String> features, IClassProbability[] classProbabilities, Map<String, Long> explanationData) {
        this(features, classProbabilities, explanationData, false, false, 0);
    }

    /**
     * @return the classProbabilities
     */
    @Override
    public IClassProbability[] getClassProbabilities() {
        return classProbabilities;
    }

    /**
     * @return the explanation
     */
    @Override
    public Map<String, Long> getExplanationData() {
        return explanationData;
    }

    /**
     * @return the laplaceSmoothed
     */
    @Override
    public boolean isLaplaceSmoothed() {
        return laplaceSmoothed;
    }

    /**
     * @return the laplaceSmoothedVariant
     */
    @Override
    public boolean isLaplaceSmoothedVariant() {
        return laplaceSmoothedVariant;
    }

    /**
     * @return the features
     */
    public Map<String, String> getFeatures() {
        return features;
    }

    /**
     * @return the laplaceSmoothingAlpha
     */
    public double getLaplaceSmoothingAlpha() {
        return laplaceSmoothingAlpha;
    }
    

}
