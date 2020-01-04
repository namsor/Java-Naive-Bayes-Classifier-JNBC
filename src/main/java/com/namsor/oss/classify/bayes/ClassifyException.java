package com.namsor.oss.classify.bayes;

/**
 * Classification exception (except persistence related).
 *
 * @author elian carsenat, NamSor SAS
 */
public class ClassifyException extends Exception {

    public ClassifyException(String message, Throwable cause) {
        super(message, cause);
    }

    public ClassifyException(String message) {
        super(message);
    }

    public ClassifyException(Throwable cause) {
        super(cause);
    }

}
