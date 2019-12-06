/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

import com.google.common.primitives.Longs;
import org.rocksdb.RocksDB;
import org.rocksdb.Options;

import java.io.*;

import java.util.Arrays;
import java.util.Set;
import java.util.logging.Logger;
import org.rocksdb.CompressionType;
import org.rocksdb.ReadOptions;
import org.rocksdb.RocksDBException;
import org.rocksdb.RocksIterator;
import org.rocksdb.WriteBatch;
import org.rocksdb.WriteOptions;

/**
 * Implementation with RocksDB
 *
 * @author elian carsenat, NamSor SAS
 */
public class NaiveBayesClassifierRocksDBImpl extends AbstractNaiveBayesClassifierImpl implements INaiveBayesClassifier {

    private final String rootPathWritable;
    private final RocksDB db;

    public NaiveBayesClassifierRocksDBImpl(String classifierName, String[] categories, String rootPathWritable, int cacheSizeMb) throws IOException, PersistentClassifierException {
        super(classifierName, categories);
        this.rootPathWritable = rootPathWritable;
        Options options = new Options();
        options.setCreateIfMissing(true);
        options.setCompressionType(CompressionType.SNAPPY_COMPRESSION);
        try {
            db = RocksDB.open(options, rootPathWritable + "/" + classifierName);
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        }
    }

    public String dbStatus() throws PersistentClassifierException {
        try {
            return db.getProperty("leveldb.stats");
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        }
    }

    public void dbClose() throws PersistentClassifierException {
        db.close();
    }

    public void dbCloseAndDestroy() throws PersistentClassifierException {
        try {
            db.close();
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

    @Override
    public synchronized void learn(String category, Set<String> features, int weight) throws ClassifyException {
        ReadOptions ro = new ReadOptions();
        ro.setSnapshot(db.getSnapshot());
        WriteOptions wo = new WriteOptions();
        WriteBatch batch = new WriteBatch();
        try {
            //private Map<K, Map<T, Counter>> featureCountPerCategory;
            batch.put(bytes(KEY_GLOBAL), Longs.toByteArray((db.get(ro, bytes(KEY_GLOBAL)) == null ? weight : Longs.fromByteArray(db.get(ro, bytes(KEY_GLOBAL))) + weight)));
            batch.put(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category), Longs.toByteArray((db.get(ro, bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category)) == null ? weight : Longs.fromByteArray(db.get(ro, bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category))) + weight)));
            for (String feature : features) {
                batch.put(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), Longs.toByteArray((db.get(ro, bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature)) == null ? weight : Longs.fromByteArray(db.get(ro, bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature))) + weight)));
                batch.put(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), Longs.toByteArray((db.get(ro, bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature)) == null ? weight : Longs.fromByteArray(db.get(ro, bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature))) + weight)));
            }
            db.write(wo, batch);
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        } finally {
            // Make sure you close the batch to avoid resource leaks.
            batch.close();
            ro.snapshot().close();
        }
    }

    @Override
    public IClassification[] classify(Set<String> features) throws ClassifyException {
        IClassification[] result = new ClassificationImpl[getCategories().length];
        ReadOptions ro = new ReadOptions();
        ro.setSnapshot(db.getSnapshot());

        //int gets = 0;
        long sM = System.currentTimeMillis();
        try {
            // return ((double) this.getCategoryCount(category) / (double) this.getCategoriesTotal()) * featuresProbabilityProduct(features, category);
            long globalCount = (db.get(ro, bytes(KEY_GLOBAL)) == null ? 0 : Longs.fromByteArray(db.get(ro, bytes(KEY_GLOBAL))));
            //gets++;
            long categoryCountTot = 0;
            for (int i = 0; i < getCategories().length; i++) {
                String category = getCategories()[i];
                long categoryCount = (db.get(ro, bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category)) == null ? 0 : Longs.fromByteArray(db.get(ro, bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category))));
                //gets++;
                double product = 1.0d;
                double weight = 1.0d;
                double assumedProbability = 1d;
                for (String feature : features) {
                    //product *= this.featureWeighedAverage(feature, category); //this.featureWeighedAverage(feature, category, null, 1.0d, 0.5d);
                    long featureCount = (db.get(ro, bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature)) == null ? 0 : Longs.fromByteArray(db.get(ro, bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature))));
                    //gets++;
                    long featureCategoryCount = 0;
                    // optim
                    if (featureCount > 0) {
                        featureCategoryCount = (db.get(ro, bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature)) == null ? 0 : Longs.fromByteArray(db.get(ro, bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature))));
                        //gets++;
                    }
                    double basicProbability = (featureCount == 0 ? 0 : 1d * featureCategoryCount / featureCount);

                    double featureWeighedAverage = (weight * assumedProbability + featureCount * basicProbability) / (weight + featureCount);
                    Logger.getLogger(getClass().getName()).info("\n\t\tcategory:" + category + " feature:" + feature + " featureCount=" + featureCount + " featureCategoryCount=" + featureCategoryCount + " basicProbability=" + basicProbability + " featureWeighedAverage=" + featureWeighedAverage);
                    product *= featureWeighedAverage;
                }
                double proba = 1d * categoryCount / globalCount * product;
                Logger.getLogger(getClass().getName()).info("\n\tcategory:" + category + " categoryCount=" + categoryCount + " globalCount=" + globalCount + " product=" + product + " proba=" + proba);
                IClassification classif = new ClassificationImpl(category, proba); // return ((double) this.getCategoryCount(category) / (double) this.getCategoriesTotal()) * featuresProbabilityProduct(features, category);
                result[i] = classif;
                categoryCountTot += categoryCount;
            }
            if (globalCount == categoryCountTot) {
                // ok we're consistent
            } else {
                // there is a slight inconsistency bw/ globalCount and categoryCount ... should not be, but not a big deal
                // throw new ClassifyException("Inconsistency : globalcount="+globalCount+" categoryCountTot="+categoryCountTot);
            }
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        } finally {
            // Make sure you close the snapshot to avoid resource leaks.
            ro.snapshot().close();
        }
        //long fM = System.currentTimeMillis();
        //Logger.getLogger(getClass().getName()).info(getClassifierName() + " cats=" + this.getCategories().length + " features=" + features.size() + " did " + gets + " get in (" + (fM - sM) + " millis.");
        Arrays.sort(result, orderByProba);
        return result;
    }

    public void dumpDb(Writer w) throws ClassifyException {
        ReadOptions ro = new ReadOptions();
        ro.setSnapshot(db.getSnapshot());

        RocksIterator iterator = db.newIterator(ro);
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

    private byte[] bytes(String key) {
        return key.getBytes();
    }
}
