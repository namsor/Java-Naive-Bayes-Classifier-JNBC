package com.namsor.oss.classify.bayes;

import java.util.Map;

/**
 * An immutable classification object
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
    private final double likelyhoodTot;
    private final boolean logProductVariant; 

    public ClassificationImpl(Map<String, String> features, IClassProbability[] classProbabilities, Map<String, Long> explanationData, boolean laplaceSmoothed, boolean laplaceSmoothedVariant, double laplaceSmoothingAlpha, double likelyhoodTot, boolean logProductVariant) {
        this.features = features;
        this.classProbabilities = classProbabilities;
        this.explanationData = explanationData;
        this.laplaceSmoothed = laplaceSmoothed;
        this.laplaceSmoothedVariant = laplaceSmoothedVariant;
        this.laplaceSmoothingAlpha = laplaceSmoothingAlpha;
        this.likelyhoodTot = likelyhoodTot; 
        this.logProductVariant=logProductVariant;
    }

    public ClassificationImpl(Map<String, String> features, IClassProbability[] classProbabilities, Map<String, Long> explanationData, double likelyhoodTot) {
        this(features, classProbabilities, explanationData, false, false, 0, likelyhoodTot, false);
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
    @Override
    public Map<String, String> getFeatures() {
        return features;
    }

    /**
     * @return the laplaceSmoothingAlpha
     */
    @Override
    public double getLaplaceSmoothingAlpha() {
        return laplaceSmoothingAlpha;
    }

    @Override
    public boolean isUnderflow() {
        return likelyhoodTot<=0;
    }

    /**
     * @return the logProductVariant
     */
    public boolean isLogProductVariant() {
        return logProductVariant;
    }


}
