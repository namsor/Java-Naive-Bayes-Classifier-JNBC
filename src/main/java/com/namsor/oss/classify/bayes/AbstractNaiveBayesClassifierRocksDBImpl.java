package com.namsor.oss.classify.bayes;

import com.google.common.primitives.Longs;
import java.io.IOException;
import java.io.Writer;
import org.rocksdb.CompressionOptions;
import org.rocksdb.CompressionType;
import org.rocksdb.Options;
import org.rocksdb.ReadOptions;
import org.rocksdb.RocksDB;
import org.rocksdb.RocksDBException;
import org.rocksdb.RocksIterator;

/**
 * Persistence methods
 * @author elian
 */
public abstract class AbstractNaiveBayesClassifierRocksDBImpl extends AbstractNaiveBayesClassifierImpl {

    private final String rootPathWritable;
    private final RocksDB db;
    private final int ROCKSDB_MaxBackgroundFlushes = 8;
    public AbstractNaiveBayesClassifierRocksDBImpl(String classifierName, String[] categories, String rootPathWritable, int topN) throws IOException, PersistentClassifierException {
        super(classifierName, categories, topN);
        this.rootPathWritable = rootPathWritable;
        Options options = new Options();
        options.setCreateIfMissing(true);
        options.setMaxBackgroundFlushes(ROCKSDB_MaxBackgroundFlushes);
        try {
            db = RocksDB.open(options, rootPathWritable + "/" + classifierName);
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        }
    }
    public AbstractNaiveBayesClassifierRocksDBImpl(String classifierName, String[] categories, String rootPathWritable) throws IOException, PersistentClassifierException {
        this(classifierName, categories, rootPathWritable, -1);
    }

    
    public long dbSize() throws PersistentClassifierException {
        try {
            String dbSize = getDb().getProperty("rocksdb.estimate-num-keys");
            if( dbSize != null) {
                return Long.parseLong(dbSize);
            } else {
                return -1;
            }
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        }
    }    
    
    @Override
    public void dbClose() throws PersistentClassifierException {
        getDb().close();
    }

    @Override
    public void dbCloseAndDestroy() throws PersistentClassifierException {
        try {
            getDb().close();
            Options options = new Options();
            RocksDB.destroyDB(rootPathWritable + "/" + getClassifierName(), options);
        } catch (RocksDBException ex) {
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
    public RocksDB getDb() {
        return db;
    }
    

    public void dumpDb(Writer w) throws ClassifyException {
        ReadOptions ro = new ReadOptions();
        ro.setSnapshot(getDb().getSnapshot());

        RocksIterator iterator = getDb().newIterator(ro);
        try {
            for (iterator.seekToFirst(); iterator.isValid(); iterator.next()) {
                String key = new String(iterator.key());
                long value = Longs.fromByteArray(iterator.value());
                w.append(key + "|" + value + "\n");
            }
        } catch (IOException ex) {
            throw new ClassifyException(ex);
        } finally {
            // Make sure you close the snapshot to avoid resource leaks.
            ro.snapshot().close();
        }
    }
    
}
