/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

import java.util.Set;
import java.util.SortedSet;

/**
 * Naive Bayes Classifier interface
 * @author elian carsenat, NamSor SAS
 */
public interface INaiveBayesClassifier {
    
    /**
     * Learn
     * @param category
     * @param features 
     */
    void learn(String category, Set<String> features) throws ClassifyException;

    /**
     * Learn
     * @param category
     * @param features 
     */
    void learn(String category, Set<String> features, int weight) throws ClassifyException;
    
    /**
     * Forget
     * @param category
     * @param features 
     */
    void forget(String category, Set<String> features) throws ClassifyException;
    
    /**
     * Predict most probable class
     * @param features 
     * @return 
     */
    IClassification[] classify(Set<String> features) throws ClassifyException;
    
    /**
     * This classifier has an immutable list of categories
     * @return 
     */
    String[] getCategories();
}
