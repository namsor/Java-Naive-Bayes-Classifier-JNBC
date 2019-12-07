package com.namsor.oss.classify.bayes;

/**
 * Classification and probability
 * @author elian carsenat, NamSor SAS
 */
public class ClassificationImpl implements IClassification {

    private final String category;
    private final double probability;

    public ClassificationImpl(String category, double probability) {
        this.category = category;
        this.probability = probability;
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

}
