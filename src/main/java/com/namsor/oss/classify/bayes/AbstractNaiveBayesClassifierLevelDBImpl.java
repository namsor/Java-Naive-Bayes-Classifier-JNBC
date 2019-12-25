package com.namsor.oss.classify.bayes;

import com.google.common.primitives.Longs;
import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.util.Map;
import static org.iq80.leveldb.impl.Iq80DBFactory.*;
//import static org.fusesource.leveldbjni.JniDBFactory.*;
import org.iq80.leveldb.CompressionType;
import org.iq80.leveldb.DB;
import org.iq80.leveldb.DBIterator;
import org.iq80.leveldb.Options;
import org.iq80.leveldb.ReadOptions;

/**
 * Persistence methods : you can switch between using 
 * - the JNI Level DB implementation (import static org.fusesource.leveldbjni.JniDBFactory.*;) or 
 * - the pure Java port (import static org.iq80.leveldb.impl.Iq80DBFactory.*;)
 *
 * @author elian
 */
public abstract class AbstractNaiveBayesClassifierLevelDBImpl extends AbstractNaiveBayesClassifierImpl {

    private static final int CACHE_SIZE_DEFAULT = 100; //100mb
    private final String rootPathWritable;
    private final DB db;

    public AbstractNaiveBayesClassifierLevelDBImpl(String classifierName, String[] categories, int cacheSizeMb, String rootPathWritable, int topN) throws PersistentClassifierException {
        super(classifierName, categories, topN);
        this.rootPathWritable = rootPathWritable;
        Options options = new Options();
        options.createIfMissing(true);
        options.cacheSize(cacheSizeMb * 1048576); // 100MB cache
        options.compressionType(CompressionType.NONE);
        
        try {
            db = factory.open(new File(rootPathWritable + "/" + classifierName), options);
        } catch (IOException ex) {
            throw new PersistentClassifierException(ex);
        }
    }

    public AbstractNaiveBayesClassifierLevelDBImpl(String classifierName, String[] categories, String rootPathWritable, int topN) throws PersistentClassifierException {
        this(classifierName, categories, CACHE_SIZE_DEFAULT, rootPathWritable, topN);
    }

    public AbstractNaiveBayesClassifierLevelDBImpl(String classifierName, String[] categories, int cacheSizeMb, String rootPathWritable) throws PersistentClassifierException {
        this(classifierName, categories, cacheSizeMb, rootPathWritable, -1);
    }

    public AbstractNaiveBayesClassifierLevelDBImpl(String classifierName, String[] categories, String rootPathWritable) throws PersistentClassifierException {
        this(classifierName, categories, CACHE_SIZE_DEFAULT, rootPathWritable, -1);
    }

    @Override
    public long dbSize() throws PersistentClassifierException {
        long dbSize = 0;
        ReadOptions ro = new ReadOptions();
        ro.snapshot(getDb().getSnapshot());
        DBIterator iterator = getDb().iterator(ro);
        try {
            for (iterator.seekToFirst(); iterator.hasNext(); iterator.next()) {
                Map.Entry<byte[],byte[]> nextEntry = iterator.peekNext();
                dbSize++;
            }
            return dbSize;
        } catch (Throwable ex) {
            throw new PersistentClassifierException(ex);
        } finally {
            try {
                // Make sure you close the snapshot to avoid resource leaks.
                ro.snapshot().close();
            } catch (IOException ex) {
                throw new PersistentClassifierException(ex);
            }
        }
    }

    @Override
    public void dbClose() throws PersistentClassifierException {
        try {
            getDb().close();
        } catch (IOException ex) {
            throw new PersistentClassifierException(ex);
        }
    }

    @Override
    public void dbCloseAndDestroy() throws PersistentClassifierException {
        try {
            db.close();
            Options options = new Options();
            factory.destroy(new File(rootPathWritable + "/" + getClassifierName()), options);
        } catch (IOException ex) {
            throw new PersistentClassifierException(ex);
        }
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        try {
            dbClose();
        } catch (Throwable t) {
            // ignore
        }
    }

    protected byte[] bytes(String key) {
        return key.getBytes();
    }

    /**
     * @return the db
     */
    public DB getDb() {
        return db;
    }

    @Override
    public synchronized void dumpDb(Writer w) throws ClassifyException {
        ReadOptions ro = new ReadOptions();
        ro.snapshot(getDb().getSnapshot());
        DBIterator iterator = getDb().iterator(ro);
        try {
            for (iterator.seekToFirst(); iterator.hasNext(); iterator.next()) {
                Map.Entry<byte[],byte[]> nextEntry = iterator.peekNext();
                String key = asString(nextEntry.getKey());
                long value = Longs.fromByteArray(nextEntry.getValue());
                w.append(key + "|" + value + "\n");
            }
        } catch (IOException ex) {
            throw new ClassifyException(ex);
        } finally {
            try {
                // Make sure you close the snapshot to avoid resource leaks.
                ro.snapshot().close();
            } catch (IOException ex) {
                throw new ClassifyException(ex);
            }
        }
    }
}
