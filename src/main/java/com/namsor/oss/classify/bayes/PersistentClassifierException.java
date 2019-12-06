/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

/**
 * Exception raised by the KeyValue backend
 * @author elian
 */
public class PersistentClassifierException extends ClassifyException {
    public PersistentClassifierException(String message, Throwable cause) {
        super(message, cause);
    }

    public PersistentClassifierException(String message) {
        super(message);
    }

    public PersistentClassifierException(Throwable cause) {
        super(cause);
    }
}
