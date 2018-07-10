/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
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

}
