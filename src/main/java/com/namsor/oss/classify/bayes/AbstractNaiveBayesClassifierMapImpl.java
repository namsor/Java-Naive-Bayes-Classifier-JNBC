/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.iq80.leveldb.Options;
import static org.iq80.leveldb.impl.Iq80DBFactory.factory;
import org.iq80.leveldb.util.FileUtils;
import org.mapdb.DB;
import org.mapdb.DBMaker;
import org.mapdb.HTreeMap;
import org.mapdb.Serializer;

/**
 * 
 * @author elian
 */
public abstract class AbstractNaiveBayesClassifierMapImpl extends AbstractNaiveBayesClassifierImpl {

    private final Map<String, Long> db;
    private final String rootPathWritable;
    private final HTreeMap<String, Long> dbPersistent;

    /**
     * Create in-memory ConcurrentHashMap classifier
     * @param classifierName
     * @param categories
     * @param topN 
     */
    public AbstractNaiveBayesClassifierMapImpl(String classifierName, String[] categories, int topN) {
        super(classifierName, categories, topN);
        this.rootPathWritable = null;
        this.dbPersistent = null;
        this.db = new ConcurrentHashMap();
    }

    /**
     * Create in-memory ConcurrentHashMap classifier
     * @param classifierName
     * @param categories 
     */
    public AbstractNaiveBayesClassifierMapImpl(String classifierName, String[] categories) {
        this(classifierName, categories, -1);
    }
    
    /**
     * Create persistent org.mapdb.HTreeMap classifier
     * @param classifierName
     * @param categories
     * @param topN
     * @param rootPathWritable 
     */
    public AbstractNaiveBayesClassifierMapImpl(String classifierName, String[] categories, int topN, String rootPathWritable) {
        super(classifierName, categories, topN);
        this.rootPathWritable = rootPathWritable;
        File dataDir = new File(rootPathWritable);
        dataDir.mkdirs();
        DB dbMap = DBMaker
                .fileDB(rootPathWritable + "/" + classifierName + ".db")
                .fileMmapEnable()
                .make();
        HTreeMap<String, Long> map = dbMap
                .hashMap("map", Serializer.STRING, Serializer.LONG)
                .createOrOpen();
        db = map;
        dbPersistent = map;
    }
    
    /**
     * Create persistent org.mapdb.HTreeMap classifier
     * @param classifierName
     * @param categories
     * @param rootPathWritable 
     */
    public AbstractNaiveBayesClassifierMapImpl(String classifierName, String[] categories, String rootPathWritable) {
        this(classifierName, categories, -1, rootPathWritable);
    }    


    @Override
    public void dbClose() throws PersistentClassifierException {
        if (dbPersistent != null) {
            dbPersistent.close();
        }
    }

    @Override
    public void dbCloseAndDestroy() throws PersistentClassifierException {
        dbClose();
        if (rootPathWritable != null) {
            File dirFile = new File(rootPathWritable);
            boolean deleted = FileUtils.deleteRecursively(dirFile);
            if( !deleted ) {
               Logger.getLogger(getClass().getName()).warning("Could not delete directory "+dirFile.getAbsolutePath());
            }
        }
    }

    @Override
    public long dbSize() throws PersistentClassifierException {
        return getDb().size();
    }

    @Override
    public synchronized void dumpDb(Writer w) throws ClassifyException {
        for (Map.Entry<String, Long> entry : getDb().entrySet()) {
            String key = entry.getKey();
            long value = entry.getValue();
            try {
                w.append(key + "|" + value + "\n");
            } catch (IOException ex) {
                Logger.getLogger(NaiveBayesClassifierMapImpl.class.getName()).log(Level.SEVERE, null, ex);
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

    /**
     * @return the rootPathWritable
     */
    public String getRootPathWritable() {
        return rootPathWritable;
    }
}
