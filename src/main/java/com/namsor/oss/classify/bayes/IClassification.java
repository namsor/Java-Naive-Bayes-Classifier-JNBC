/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

/**
 * Classification output
 * @author elian carsenat, NamSor SAS
 */
public interface IClassification {
    /**
     * Category
     * @return 
     */
    String getCategory();
    /**
     * Probability
     * @return 
     */
    double getProbability();
}
