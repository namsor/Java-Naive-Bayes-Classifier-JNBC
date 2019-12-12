package com.namsor.oss.classify.bayes;

import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.rocksdb.CompressionType;
import org.rocksdb.Options;
import org.rocksdb.RocksDB;
import org.rocksdb.RocksDBException;

/**
 * Persistence methods
 * @author elian
 */
public abstract class AbstractNaiveBayesClassifierRocksDBImpl extends AbstractNaiveBayesClassifierImpl {

    private final String rootPathWritable;
    private final RocksDB db;

    public AbstractNaiveBayesClassifierRocksDBImpl(String classifierName, String[] categories, String rootPathWritable) throws IOException, PersistentClassifierException {
        super(classifierName, categories);
        this.rootPathWritable = rootPathWritable;
        Options options = new Options();
        options.setCreateIfMissing(true);
        options.setCompressionType(CompressionType.NO_COMPRESSION);
        try {
            db = RocksDB.open(options, rootPathWritable + "/" + classifierName);
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        }
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
    
}
