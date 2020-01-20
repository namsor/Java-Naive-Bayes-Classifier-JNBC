package com.namsor.oss.classify.bayes;

import com.namsor.oss.classify.bayes.ClassifyException;

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
