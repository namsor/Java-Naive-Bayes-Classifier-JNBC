package com.namsor.oss.classify.bayes;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.iq80.leveldb.util.FileUtils;
import org.mapdb.DB;
import org.mapdb.DBMaker;
import org.mapdb.HTreeMap;
import org.mapdb.Serializer;

/**
 * A simple, scalable Naive Bayes Classifier, based on a key-value store (in
 * memory using ConcurrentHashMap, or disk-based using org.mapdb.HTreeMap)
 * @author elian
 */
public abstract class AbstractNaiveBayesClassifierMapImpl extends AbstractNaiveBayesClassifierImpl {

    private final Map<String, Long> db;
    private final String rootPathWritable;
    private final HTreeMap<String, Long> dbPersistent;

    /**
     * Create in-memory classifier using ConcurrentHashMap
     * @param classifierName The classifier name
     * @param categories The classification categories
     */
    public AbstractNaiveBayesClassifierMapImpl(String classifierName, String[] categories) {
        super(classifierName, categories);
        this.rootPathWritable = null;
        this.dbPersistent = null;
        this.db = new ConcurrentHashMap();
    }

    /**
     * Create persistent classifier using org.mapdb.HTreeMap 
     * @param classifierName The classifier name
     * @param categories The classification categories
     * @param rootPathWritable A writable directory for org.mapdb.HTreeMap storage
     */
    public AbstractNaiveBayesClassifierMapImpl(String classifierName, String[] categories, String rootPathWritable) {
        super(classifierName, categories);
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
    
    @Override
    public void dbClose() throws PersistentClassifierException { //todo PersistentClassifierException is never thrown
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
    public long dbSize() throws PersistentClassifierException { // todo PersistentClassifierException is never thrown
        return getDb().size();
    }

    @Override
    public synchronized void dumpDb(Writer w) throws PersistentClassifierException {
        for (Map.Entry<String, Long> entry : getDb().entrySet()) {
            String key = entry.getKey();
            long value = entry.getValue();
            try {
                w.append(key + "|" + value + "\n");
            } catch (IOException ex) {
                Logger.getLogger(NaiveBayesClassifierMapImpl.class.getName()).log(Level.SEVERE, null, ex);
                throw new PersistentClassifierException(ex);
            }
        }
    }

    /**
     * @return the db
     */
    protected Map<String, Long> getDb() {
        return db;
    }

    /**
     * @return the rootPathWritable
     */
    public String getRootPathWritable() {
        return rootPathWritable;
    }
}
