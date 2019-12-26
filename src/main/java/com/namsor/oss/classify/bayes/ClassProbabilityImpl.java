package com.namsor.oss.classify.bayes;

/**
 * Classification output and probability estimate.
 * @author elian carsenat, NamSor SAS
 */
public class ClassProbabilityImpl implements IClassProbability {

    private final String category;
    private final double probability;
    private final boolean other;

    public ClassProbabilityImpl(String category, double probability) {
        this.category = category;
        this.probability = probability;
        this.other = false;
    }

    public ClassProbabilityImpl(String category, double probability, boolean other) {
        this.category = category;
        this.probability = probability;
        this.other = other;
    }
    
    @Override
    public String getCategory() {
        return category;
    }

    @Override
    public double getProbability() {
        return probability;
    }

    @Override
    public String toString() {
        return "P(" + category + ")=" + probability;
    }


    @Override
    public boolean isOther() {
        return other;
    }

}
