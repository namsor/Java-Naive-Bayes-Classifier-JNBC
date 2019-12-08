/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

import com.google.common.primitives.Longs;
import static com.namsor.oss.classify.bayes.AbstractNaiveBayesClassifierImpl.pathCategory;
import static com.namsor.oss.classify.bayes.AbstractNaiveBayesClassifierImpl.pathGlobal;
import org.rocksdb.RocksDB;
import org.rocksdb.Options;

import java.io.*;

import java.util.Arrays;
import java.util.Map;
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

    public NaiveBayesClassifierRocksDBImpl(String classifierName, String[] categories, String rootPathWritable) throws IOException, PersistentClassifierException {
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

    public String dbStatus() throws PersistentClassifierException {
        try {
            return db.getProperty("leveldb.stats");
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        }
    }

    @Override
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
    public synchronized void learn(String category, Map<String, String> features, long weight) throws ClassifyException {
        ReadOptions ro = new ReadOptions();
        ro.setSnapshot(db.getSnapshot());
        WriteOptions wo = new WriteOptions();
        WriteBatch batch = new WriteBatch();
        try {

            String pathGlobal = pathGlobal();
            batch.put(bytes(pathGlobal), Longs.toByteArray((db.get(ro, bytes(pathGlobal)) == null ? weight : Longs.fromByteArray(db.get(ro, bytes(pathGlobal))) + weight)));
            String pathCategory = pathCategory(category);
            batch.put(bytes(pathCategory), Longs.toByteArray((db.get(ro, bytes(pathCategory)) == null ? weight : Longs.fromByteArray(db.get(ro, bytes(pathCategory))) + weight)));
            for (Map.Entry<String, String> feature : features.entrySet()) {
                String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                batch.put(bytes(pathCategoryFeatureKey), Longs.toByteArray((db.get(ro, bytes(pathCategoryFeatureKey)) == null ? weight : Longs.fromByteArray(db.get(ro, bytes(pathCategoryFeatureKey))) + weight)));
                String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                batch.put(bytes(pathCategoryFeatureKeyValue), Longs.toByteArray((db.get(ro, bytes(pathCategoryFeatureKeyValue)) == null ? weight : Longs.fromByteArray(db.get(ro, bytes(pathCategoryFeatureKeyValue))) + weight)));
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
    public IClassification[] classify(Map<String, String> features) throws ClassifyException {
        IClassification[] result = new ClassificationImpl[getCategories().length];
        ReadOptions ro = new ReadOptions();
        ro.setSnapshot(db.getSnapshot());
        try {
            String pathGlobal = pathGlobal();
            long globalCount = (db.get(ro, bytes(pathGlobal)) == null ? 0 : Longs.fromByteArray(db.get(ro, bytes(pathGlobal))));
            double[] likelyhood = new double[getCategories().length];
            double likelyhoodTot = 0;
            for (int i = 0; i < getCategories().length; i++) {
                String category = getCategories()[i];
                String pathCategory = pathCategory(category);
                long categoryCount = (db.get(ro, bytes(pathCategory)) == null ? 0 : Longs.fromByteArray(db.get(ro, bytes(pathCategory))));
                double product = 1.0d;
                for (Map.Entry<String, String> feature : features.entrySet()) {
                    String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                    double featureCount = (db.get(ro, bytes(pathCategoryFeatureKey)) == null ? 0 : Longs.fromByteArray(db.get(ro, bytes(pathCategoryFeatureKey))));
                    String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                    double featureCategoryCount = (db.get(ro, bytes(pathCategoryFeatureKeyValue)) == null ? 0 : Longs.fromByteArray(db.get(ro, bytes(pathCategoryFeatureKeyValue))));
                    double basicProbability = (featureCount == 0 ? 0 : 1d * featureCategoryCount / featureCount);
                    product *= basicProbability;
                }
                double proba = 1d * categoryCount / globalCount * product;
                likelyhood[i] = 1d * categoryCount / globalCount * product;
                likelyhoodTot += likelyhood[i];
            }
            for (int i = 0; i < getCategories().length; i++) {
                double proba = likelyhood[i] / likelyhoodTot;
                ClassificationImpl classif = new ClassificationImpl(getCategories()[i], proba);
                result[i] = classif;
            }
            Arrays.sort(result, orderByProba);
            return result;
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        } finally {
            // Make sure you close the snapshot to avoid resource leaks.
            ro.snapshot().close();
        }

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
