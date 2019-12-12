/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

import java.io.IOException;
import java.io.Writer;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author elian
 */
public abstract class AbstractNaiveBayesClassifierTransientImpl extends AbstractNaiveBayesClassifierImpl {

    private final Map<String, Long> db;

    public AbstractNaiveBayesClassifierTransientImpl(String classifierName, String[] categories) {
        super(classifierName, categories);
        db = new ConcurrentHashMap();
    }

    @Override
    public void dbClose() throws PersistentClassifierException {
    }

    @Override
    public void dbCloseAndDestroy() throws PersistentClassifierException {
    }

    @Override
    public long dbSize() throws PersistentClassifierException {
        return getDb().size();
    }

    public synchronized void dumpDb(Writer w) throws ClassifyException {
        for (Map.Entry<String, Long> entry : getDb().entrySet()) {
            String key = entry.getKey();
            long value = entry.getValue();
            try {
                w.append(key + "|" + value + "\n");
            } catch (IOException ex) {
                Logger.getLogger(NaiveBayesClassifierTransientImpl.class.getName()).log(Level.SEVERE, null, ex);
                throw new ClassifyException(ex);
            }
        }
    }

    /**
     * @return the db
     */
    public Map<String, Long> getDb() {
        return db;
    }
}
